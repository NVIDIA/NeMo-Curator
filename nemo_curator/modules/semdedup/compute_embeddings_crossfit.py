import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
import yaml

import pdb
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
    return torch.sum(token_embeddings * input_mask_expanded, 1)/torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config.model, config=self.config)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        with torch.autocast(device_type=input_ids.device.type):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return embeddings

    @torch.no_grad()
    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        emb = mean_pooling(feature, batch["attention_mask"])
        return F.normalize(emb, dim=1)


def load_model(config, device):
    model = CustomModel(config)
    model = model.to(device)
    model.eval()
    return model


class MyModel(HFModel):
    def __init__(self, config):
        self.config = config
        super().__init__(self.config.model)

    def load_model(self, device="cuda"):
        return load_model(self.config, device=device)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)

    def max_seq_length(self):
        return self.config.max_seq_length


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PyTorch Model Predictions using Crossfit"
    )
    parser.add_argument(
        "--config_file", help="YAML with configs",
        default="configs_cf.yml")

    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    return params


def read_json(paths):
    return cudf.read_json(paths, lines=True)


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


def find_remaining_files(input_files, out_file_dir):
    output_files = sorted([os.path.join(out_file_dir, x) for x in os.listdir(out_file_dir)])
    print ('output_files', len(output_files))
    print ('input_files', len(input_files))

    in_ext = os.path.basename(input_files[0]).split('.')[-1]
    out_basenames = [os.path.basename(f).replace('.parquet','')  for f in output_files]
    in_basenames = [os.path.basename(f).replace(in_ext, '')[:-1] for f in input_files]

    print ('in_basenames', len(in_basenames))
    print ('out_basenames', len(out_basenames))

    remaining = [f for f in in_basenames if f not in out_basenames]
    in_dirname = os.path.dirname(input_files[0])
    print('remaining', len(remaining))
    print ('in_dirname', in_dirname)
    return [f'{in_dirname}/{f}.{in_ext}' for f in remaining]


def main():
    args = parse_arguments()
    args_emb = args['embeddings']

    @dataclass
    class Config:
        model: str = args_emb['model_name']
        max_len_to_use = AutoTokenizer.from_pretrained('facebook/opt-125m').model_max_length
        if max_len_to_use > 1e5:
            max_len_to_use = AutoConfig.from_pretrained(model).max_position_embeddings
        max_seq_length: int = max_len_to_use

    cluster = LocalCUDACluster(
        rmm_pool_size=args_emb['pool_size'], n_workers=args_emb['num_workers'],rmm_async=True
    )

    client = Client(cluster)

    st = time.time()

    data_path = f"{args['root']}/{args_emb['datapath']}"
    input_files = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])
    input_files = input_files[:args['sample']]
    out_file_dir = f"{args['root']}/{args_emb['emb_parquet_path']}"
    os.makedirs(out_file_dir, exist_ok=True)
    input_files = find_remaining_files(input_files, out_file_dir)
    if len(input_files) == 0:
        print ('No more input files to generate embeddings')
        return
    print(Config.max_seq_length)
    ddf = dask_cudf.read_json(input_files, lines=True, include_path_column=True)
    model = MyModel(Config)
    pipe = op.Sequential(
        op.Tokenizer(
            model,
            cols=[args_emb['input_column']],
            tokenizer_type="sentencepiece",
            max_length=Config.max_seq_length,
        ),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=args_emb['batch_size'],
            pred_output_col="embeddings",
        ),
        keep_cols=ddf.columns.tolist(),
    )

    outputs = pipe(ddf)

    outputs.map_partitions(
        single_partition_write_with_filename,
        output_file_dir=out_file_dir,
        meta=cudf.Series(dtype=bool),
    ).compute()

    print(f"Time taken: {time.time() - st}")


if __name__ == "__main__":
    main()
