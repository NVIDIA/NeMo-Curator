# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_remaining_files
from nemo_curator.utils.script_utils import add_distributed_args, parse_client_args


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return sum_embeddings / sum_mask


## Define the model config
@dataclass
class EmbeddingConfig:
    path_or_name: str
    max_seq_length: int = None

    def __post_init__(self):
        # Set max_seq_length based on model's capabilities
        self.max_seq_length = AutoTokenizer.from_pretrained(
            self.path_or_name
        ).model_max_length
        # Guard against excessively large max lengths
        if self.max_seq_length > 1e5:
            self.max_seq_length = AutoConfig.from_pretrained(
                self.path_or_name
            ).max_position_embeddings


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config.path_or_name, config=self.config)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        with torch.autocast(device_type=input_ids.device.type):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return embeddings

    @torch.no_grad()
    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        emb = mean_pooling(feature, batch["attention_mask"])
        return F.normalize(emb, dim=1)


class CrossFitModel(HFModel):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        super().__init__(self.config.path_or_name, max_mem_gb=28)

    def load_model(self, device="cuda"):
        model = CustomModel(self.config)
        model = model.to(device)
        model.eval()
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)

    def max_seq_length(self):
        return self.config.max_seq_length


def merge_args_with_config(args, config):
    for key, value in config.items():
        if not getattr(args, key, None):
            print(f"Setting {key} to {value}")
            setattr(args, key, value)
    return args


def parse_arguments():
    parser = ArgumentParser(description="PyTorch Model Predictions using Crossfit")

    parser = add_distributed_args(parser)
    # Set low default RMM pool size for classifier
    # to allow pytorch to grow its memory usage
    # by default
    parser.set_defaults(rmm_pool_size="512MB")
    parser.set_defaults(device="gpu")
    parser.set_defaults(set_torch_to_use_rmm=False)
    parser.add_argument(
        "--config_file", help="YAML with configs", default="configs_cf.yml"
    )
    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as y_file:
        config_args = yaml.load(y_file, Loader=yaml.FullLoader)

    return merge_args_with_config(args, config_args)


def main():
    args = parse_arguments()
    sample = args.sample
    args_emb = args.embeddings

    st = time.time()

    input_data_dir = args_emb["input_data_dir"]
    output_data_dir = args_emb["output_data_dir"]

    os.makedirs(output_data_dir, exist_ok=True)

    len_written_files = len(os.listdir(output_data_dir))
    input_files = get_remaining_files(input_data_dir, output_data_dir, "jsonl")

    if sample > 0 and len(input_files) > sample:
        left_to_sample = 0
    else:
        left_to_sample = len(input_files) - len_written_files

    if left_to_sample == 0:
        print("No files to process")
        return
    else:
        input_files = input_files[:left_to_sample]
        print(f"Processing {left_to_sample} files")

    client = get_client(**parse_client_args(args))
    ddf = read_data(
        input_files=input_files,
        file_type="jsonl",
        add_filename=True,
    )
    embeddings_config = EmbeddingConfig(path_or_name=args_emb["path_or_name"])
    model = CrossFitModel(embeddings_config)
    pipe = op.Sequential(
        op.Tokenizer(
            model,
            cols=[args_emb["input_column"]],
            tokenizer_type="sentencepiece",
            max_length=EmbeddingConfig.max_seq_length,
        ),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=args_emb["batch_size"],
            pred_output_col="embeddings",
        ),
        keep_cols=ddf.columns.tolist(),
    )

    ddf = pipe(ddf)
    write_to_disk(
        ddf,
        output_data_dir,
        write_to_filename=True,
        output_type="parquet",
    )

    print(f"Time taken: {time.time() - st}")


if __name__ == "__main__":
    main()
