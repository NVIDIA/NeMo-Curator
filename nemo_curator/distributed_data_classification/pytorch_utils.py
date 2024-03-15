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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel


class CFG:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2

    def __init__(self, max_len=512):
        self.max_len = max_len


def collate(inputs):
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        # CPMP: no need to truncate labels
        if k != "labels":
            inputs[k] = inputs[k][:, :mask_len]
    return inputs


class CustomModel(nn.Module):
    def __init__(self, cfg, out_dim, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg.model, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, out_dim)
        self._init_weights(self.fc)

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
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.fc(self.fc_dropout(feature))
        return output


class TestDataset(Dataset):
    def __init__(self, cfg, df, max_chars):
        self.cfg = cfg
        text = df["text"].str.slice(0, max_chars).to_arrow().to_pylist()
        with torch.no_grad():
            self.tokens = cfg.tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=cfg.max_len,
                pad_to_max_length=True,
                truncation=True,
                return_token_type_ids=False,
            )
        self.max_chars = max_chars
        self.dataset_len = len(text)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.tokens.items()}
