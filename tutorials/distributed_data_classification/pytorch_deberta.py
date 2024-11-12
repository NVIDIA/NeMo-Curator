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
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoModel
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    _run_classifier_helper,
)
from nemo_curator.datasets import DocumentDataset


@dataclass
class PyTorchModelConfig:
    base_model: str = "microsoft/deberta-v3-base"
    fc_dropout: float = 0.2
    max_len: int = 512


class NCCustomModel(nn.Module):
    def __init__(
        self,
        config: dataclass,
        out_dim: int,
        config_path: str = None,
        pretrained: bool = False,
        autocast: bool = False,
    ):
        super().__init__()
        self.config = config
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                config.base_model, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(
                config.base_model, config=self.config
            )
        else:
            self.model = AutoModel(self.config)

        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, out_dim)
        self.autocast = autocast

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def _forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.fc(self.fc_dropout(feature))
        output = output.to(torch.float32)
        return torch.softmax(output[:, 0, :], dim=1)

    def forward(self, batch):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(batch)
        else:
            return self._forward(batch)


class PyTorchModel(HFModel):
    def __init__(
        self,
        config: dataclass,
        out_dim: int,
        model_path: str,
        autocast: bool = False,
    ):
        self.config = config
        self.out_dim = out_dim
        self.model_path = model_path
        self.autocast = autocast
        super().__init__(self.config.base_model)

    def load_model(self, device: str = "cuda"):
        model = NCCustomModel(
            self.config,
            out_dim=self.out_dim,
            config_path=None,
            pretrained=True,
            autocast=self.autocast,
        )
        model = model.to(device)

        if os.path.exists(self.model_path):
            sd = torch.load(self.model_path, map_location="cpu")
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
            model.load_state_dict(sd, strict=True)
        else:
            raise ValueError(f"Model path {self.model_path} does not exist")

        return model.eval()

    def load_tokenizer(self):
        # TODO: Allow user to pass in their own tokenizer if base_model is not Deberta
        return DebertaV2TokenizerFast.from_pretrained(self.config.base_model)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


class PyTorchClassifier(DistributedDataClassifier):
    """
    PyTorchClassifier is a general classifier designed for running generic PTH model files.
    This class is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        pretrained_model_name_or_path (str): The path to your PyTorch model file.
        labels (list[str]): The classes output by the model classifier.
        out_dim (list[str], optional): Set to 1 for a binary classification task. Otherwise, defaults to len(labels).
        filter_by (list[str], optional): The classes to filter the dataset by. If None, all classes will be included. Defaults to None.
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "pred".
        prob_column (str): The column name where prediction probabilities will be stored. Defaults to "prob".
        max_chars (int): The maximum number of characters in each document to consider for classification. Defaults to 6000.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        base_model (str): The base model on which your PyTorch model was trained. Defaults to "microsoft/deberta-v3-base".
        fc_dropout (str): Dropout rate used during training. Defaults to 0.2.
        max_len (str): Maximum sequence length used during training. Defaults to 512.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        labels: List[str],
        out_dim: Optional[int] = None,
        filter_by: Optional[List[str]] = None,
        batch_size: int = 256,
        text_field: str = "text",
        pred_column: str = "pred",
        prob_column: str = "prob",
        max_chars: int = 6000,
        device_type: str = "cuda",
        autocast: bool = True,
        base_model: str = "microsoft/deberta-v3-base",
        fc_dropout: float = 0.2,
        max_len: int = 512,
    ):
        config = PyTorchModelConfig(
            base_model=base_model,
            fc_dropout=fc_dropout,
            max_len=max_len,
        )

        self.labels = labels
        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = len(labels)

        self.text_field = text_field
        self.prob_column = prob_column

        model = PyTorchModel(
            config=config,
            out_dim=self.out_dim,
            model_path=pretrained_model_name_or_path,
            autocast=autocast,
        )

        super().__init__(
            model=model,
            labels=self.labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=self.out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting PyTorch classifier inference", flush=True)
        df = dataset.df
        df = _run_classifier_helper(
            df=df,
            model=self.model,
            labels=self.labels,
            max_chars=self.max_chars,
            batch_size=self.batch_size,
            label_col=self.pred_column,
            text_field=self.text_field,
            prob_col=self.prob_column,
        )
        return DocumentDataset(df)
