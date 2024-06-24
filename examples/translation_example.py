import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import time
from dataclasses import dataclass
from functools import lru_cache

import cudf
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel

# TODO: Import for IndicTransTokenizer
# Check if we want to add this as an dependency
try:
    from IndicTransTokenizer import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransTokenizer not found. Please install it using the following command: \n"
        + "pip install git+https://github.com/VarunGumma/IndicTransTokenizer.git"
    )
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from nemo_curator.utils.distributed_utils import (
    get_client,
    load_object_on_worker,
    read_data,
    write_to_disk,
)
from nemo_curator.utils.script_utils import (
    parse_client_args,
    parse_distributed_classifier_args,
)


@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    max_length: int = 256
    num_beams: int = 5
    autocast: bool = False


class CustomModel(nn.Module):
    ddef __init__(self, config: TranslationConfig, pretrained_model_name_or_path: str, autocast: bool = False):
        super().__init__()
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.autocast = autocast

    @torch.no_grad()
    def _forward(self, batch):
        return self.model.generate(
            **batch,
            use_cache=True,
            min_length=0,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            num_return_sequences=1,
        )

    def forward(self, batch):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                outputs = self._forward(batch)
        else:
            outputs = self._forward(batch)
        return outputs


class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config):
        self.trans_config = config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        super().__init__(self.trans_config.pretrained_model_name_or_path)


    def load_model(self, device="cuda"):
        model = CustomModel(
            self.trans_config, self.trans_config.pretrained_model_name_or_path, self.trans_config.autocast
        )
        model = model.to(device)
        model.eval()
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def max_seq_length(self) -> int:
        return self.config.max_source_positions

    @lru_cache(maxsize=1)
    def load_cfg(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )


def preprocess_df(df):
    ip = load_object_on_worker("IndicProcessor", IndicProcessor, {"inference": True})
    sentences = df["text"].to_arrow().to_pylist()
    sentences = ip.preprocess_batch(
        sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", show_progress_bar=False
    )
    df["text"] = cudf.Series(sentences)
    return df


def translate_tokens(df, model):
    ip = load_object_on_worker("IndicProcessor", IndicProcessor, {"inference": True})
    tokenizer = model.load_tokenizer()
    generated_tokens = df["translation"].to_arrow().to_pylist()
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            src=False,
        )
        generated_tokens = ip.postprocess_batch(generated_tokens, lang="hin_Deva")
    df["translation"] = cudf.Series(generated_tokens)
    return df


def parse_arguments():
    parser = parse_distributed_classifier_args()
    parser.add_argument(
        "--input-column",
        type=str,
        required=False,
        default="text",
        help="The column name in the input data that contains the text to be translated",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"Arguments parsed = {args}")
    client = get_client(**parse_client_args(args))
    print(client.dashboard_link)
    st = time.time()
    translation_config = TranslationConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        max_length=256,
        num_beams=5,
        autocast=args.autocast,
    )
    input_files = [
        os.path.join(args.input_data_dir, x) for x in os.listdir(args.input_data_dir)
    ]
    ddf = read_data(
        input_files,
        file_type=args.input_file_type,
        backend="cudf",
        files_per_partition=1,
        add_filename=True,
    )
    ddf = ddf.map_partitions(preprocess_df, meta=ddf._meta.copy())
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
    write_to_disk(
        ddf,
        output_file_dir=args.output_data_dir,
        write_to_filename=True,
        output_type=args.output_file_type,
    )
    print(f"Total time taken for translation: {time.time()-st} seconds", flush=True)

    client.close()


if __name__ == "__main__":
    main()

