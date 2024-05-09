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
import warnings

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import torch
from packaging import version
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.models.deberta_v2 import DebertaV2TokenizerFast
from transformers import AutoConfig, AutoModel
from dataclasses import dataclass
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel


from nemo_curator.distributed_data_classification.arg_utils import create_arg_parser
from nemo_curator.utils.distributed_utils import (
    get_client,
    read_data,
    write_to_disk,
)
from nemo_curator.utils.distributed_utils import read_data
from nemo_curator.utils.file_utils import get_remaining_files


warnings.filterwarnings("ignore")


@dataclass
class Config:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2
    max_len = 512


class CustomModel(nn.Module):
    def __init__(
        self, config, out_dim, config_path=None, pretrained=False, autocast=False
    ):
        super().__init__()
        self.config = config
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                config.model, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(config.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, out_dim)
        self._init_weights(self.fc)
        self.autocast = autocast

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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, batch):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                feature = self.feature(batch["input_ids"], batch["attention_mask"])
                output = self.fc(self.fc_dropout(feature))
                output = output.to(torch.float32)
        else:
            feature = self.feature(batch["input_ids"], batch["attention_mask"])
            output = self.fc(self.fc_dropout(feature))
        return torch.softmax(output[:, 0, :], dim=1)


def load_model(config, device, model_path, autocast):
    """
    This function loads the domain model and prepares it to be used for inference.
    It is needed as an input to the `process_all_batches` function within the `inference_per_partition` function.

    Args:
        config: A config object.
        device: A specified PyTorch device, such as torch.device("cuda") or torch.device("cpu").
        model_path: The path to the model file.
        autocast: Wether to autocast or not
    Returns:
        The loaded model.

    """
    model = CustomModel(
        config, out_dim=27, config_path=None, pretrained=True, autocast=autocast
    )
    model = model.to(device)
    if os.path.exists(model_path):
        sd = torch.load(os.path.join(model_path), map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
        if version.parse(TRANSFORMERS_VERSION) >= version.parse("4.31.0"):
            sd.pop("model.embeddings.position_ids", None)
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


class DomainModel(HFModel):
    def __init__(self, config, model_path=None, autocast=False):
        self.config = config
        self.model_path = model_path
        self.autocast = autocast
        super().__init__(self.config.model)

    def load_model(self, device="cuda"):
        return load_model(
            self.config,
            device=device,
            model_path=self.model_path or self.path_or_name,
            autocast=self.autocast,
        )

    def load_tokenizer(self):
        return DebertaV2TokenizerFast.from_pretrained(self.config.model)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


def main():
    labels = [
        "Adult",
        "Arts_and_Entertainment",
        "Autos_and_Vehicles",
        "Beauty_and_Fitness",
        "Books_and_Literature",
        "Business_and_Industrial",
        "Computers_and_Electronics",
        "Finance",
        "Food_and_Drink",
        "Games",
        "Health",
        "Hobbies_and_Leisure",
        "Home_and_Garden",
        "Internet_and_Telecom",
        "Jobs_and_Education",
        "Law_and_Government",
        "News",
        "Online_Communities",
        "People_and_Society",
        "Pets_and_Animals",
        "Real_Estate",
        "Reference",
        "Science",
        "Sensitive_Subjects",
        "Shopping",
        "Sports",
        "Travel_and_Transportation",
    ]

    args = create_arg_parser().parse_args()
    print(f"Arguments parsed = {args}", flush=True)
    max_chars = 2000
    batch_size = args.batch_size

    client = get_client(args, cluster_type="gpu")
    print("Starting domain classifier inference", flush=True)
    global_st = time.time()
    files_per_run = len(client.scheduler_info()["workers"]) * 2
    input_files = get_remaining_files(
        args.input_file_path, args.output_file_path, args.input_file_type
    )
    print(f"Total input files {len(input_files)}", flush=True)

    if args.input_file_type == "pickle":
        add_filename = False
    else:
        add_filename = True

    for file_batch_id, i in enumerate(range(0, len(input_files), files_per_run)):
        batch_st = time.time()
        current_batch_files = input_files[i : i + files_per_run]
        print(
            f"File Batch ID {file_batch_id}: total input files {len(current_batch_files)}",
            flush=True,
        )
        df = read_data(
            input_files=current_batch_files,
            file_type=args.input_file_type,
            add_filename=add_filename,
        )
        df["sliced_text"] = df["text"].str.slice(0, max_chars)
        columns_to_keep_list = df.columns.to_list()
        columns_to_keep_list.remove("sliced_text")

        model = DomainModel(
            Config, model_path=args.model_file_name, autocast=args.autocast
        )
        pipe = op.Sequential(
            op.Tokenizer(model, cols=["sliced_text"], tokenizer_type="sentencepiece"),
            op.Predictor(model, sorted_data_loader=True, batch_size=batch_size),
            op.Labeler(labels, cols=["preds"]),
            repartition=df.npartitions,
            keep_cols=columns_to_keep_list,
        )
        df = pipe(df)

        write_to_disk(
            df=df,
            output_file_dir=args.output_file_path,
            write_to_filename=add_filename,
        )
        batch_et = time.time()
        print(
            f"File Batch ID {file_batch_id}: completed in {batch_et-batch_st} seconds",
            flush=True,
        )

    global_et = time.time()
    print(
        f"Total time taken for domain classifier inference: {global_et-global_st} s",
        flush=True,
    )
    client.close()


def console_script():
    main()
