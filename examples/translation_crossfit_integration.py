import argparse
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cudf
import dask_cudf
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    max_length: int = 256
    num_beams: int = 5


class CustomModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def forward(self, batch):
        outputs = self.model.generate(
            **batch,
            use_cache=True,
            min_length=0,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            num_return_sequences=1,
        )
        return outputs


class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config):
        self.config = config
        super().__init__(config.pretrained_model_name_or_path)

    def load_model(self, device="cuda"):
        return load_model(config=self.config, device=device)

    def load_config(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def max_seq_length(self) -> int:
        return self.config.max_length

    @lru_cache(maxsize=1)
    def load_cfg(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )


def load_model(config, device):
    model = CustomModel(config.pretrained_model_name_or_path)
    model = model.to(device)
    model.eval()
    return model


def translate_tokens(df, model):
    tokenizer = model.load_tokenizer()
    generated_tokens = df["translation"].to_arrow().to_pylist()
    generated_tokens = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    df["translation"] = cudf.Series(generated_tokens)
    return df


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch Model Predictions using Crossfit"
    )
    parser.add_argument(
        "--input-jsonl-path", help="Input JSONL file path", required=True
    )
    parser.add_argument(
        "--output-parquet-path", help="Output Parquet file path", required=True
    )
    parser.add_argument(
        "--input-column",
        type=str,
        default="text",
        help="Column name in input dataframe",
    )
    parser.add_argument("--pool-size", type=str, default="1GB", help="RMM pool size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="ai4bharat/indictrans2-en-indic-1B",
        help="Model name",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    return parser.parse_args()


def single_partition_write_with_filename(df, output_file_dir):
    assert "path" in df.columns

    if len(df) > 0:
        empty_partition = True
    else:
        warnings.warn("Empty partition found")
        empty_partition = False

    success_ser = cudf.Series([empty_partition])
    if empty_partition:
        filename = df.path.iloc[0]
        num_filenames = len(df.path.unique())
        if num_filenames > 1:
            raise ValueError(
                f"More than one filename found in partition: {num_filenames}"
            )
        filename = Path(filename).stem
        output_file_path = os.path.join(output_file_dir, f"{filename}.parquet")
        df["path"] = df["path"].astype(str)
        df.to_parquet(output_file_path)

    return success_ser


def main():
    args = parse_arguments()

    cluster = LocalCUDACluster(
        rmm_pool_size=args.pool_size, n_workers=args.num_workers, rmm_async=True
    )
    client = Client(cluster)
    print(client.dashboard_link)

    translation_config = TranslationConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        max_length=256,
        num_beams=5,
    )
    input_files = [
        os.path.join(args.input_jsonl_path, x)
        for x in os.listdir(args.input_jsonl_path)
    ]
    # ddf = dask_cudf.read_json(input_files, lines=True, include_path_column=True)
    ddf = dask_cudf.read_parquet(input_files, include_path_column=True)
    columns = ddf.columns.tolist()
    model = ModelForSeq2SeqModel(translation_config)
    pipe = op.Sequential(
        op.Tokenizer(model, cols=[args.input_column], tokenizer_type="sentencepiece"),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=args.batch_size,
            pred_output_col="translation",
        ),
        keep_cols=columns,
    )
    ddf = pipe(ddf)
    translated_meta = ddf._meta.copy()
    translated_meta["translation"] = "DUMMY_STRING"
    ddf = ddf.map_partitions(translate_tokens, model=model, meta=translated_meta)

    # Create output directory if it does not exist
    os.makedirs(args.output_parquet_path, exist_ok=True)
    ddf.map_partitions(
        single_partition_write_with_filename,
        output_file_dir=args.output_parquet_path,
        meta=cudf.Series(dtype=bool),
    ).compute()


if __name__ == "__main__":
    main()
