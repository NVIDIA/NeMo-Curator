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
from dataclasses import dataclass

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoTokenizer

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    HFDeberta,
    _get_suggest_memory_for_classifier,
    _run_classifier_helper,
)
from nemo_curator.datasets import DocumentDataset

DOMAIN_IDENTIFIER = "nvidia/domain-classifier"


@dataclass
class DomainModelConfig:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2
    max_len = 512


class DomainModel(HFModel):
    def __init__(
        self, config: DomainModelConfig, autocast: bool = False, max_mem_gb=None
    ):
        self.config = config
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()

        super().__init__(self.config.model, max_mem_gb=max_mem_gb)

    def load_model(self, device="cuda"):
        model = HFDeberta.from_pretrained(DOMAIN_IDENTIFIER)
        model.set_autocast(self.autocast)
        model = model.to(device)
        return model.eval()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(DOMAIN_IDENTIFIER)

    def load_config(self):
        return AutoConfig.from_pretrained(DOMAIN_IDENTIFIER)


class DomainClassifier(DistributedDataClassifier):
    def __init__(
        self,
        filter_by=None,
        batch_size=256,
        pred_column="domain_pred",
        prob_column=None,
        max_chars=2000,
        device_type="cuda",
        autocast=True,
        max_mem_gb=None,
    ):
        config = AutoConfig.from_pretrained(DOMAIN_IDENTIFIER)

        self.prob_column = prob_column
        self.labels = list(config.label2id.keys())
        self.out_dim = len(self.labels)

        model = DomainModel(
            config=DomainModelConfig, autocast=autocast, max_mem_gb=max_mem_gb
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
        print("Starting domain classifier inference", flush=True)
        df = dataset.df
        df = _run_classifier_helper(
            df=df,
            model=self.model,
            labels=self.labels,
            max_chars=self.max_chars,
            batch_size=self.batch_size,
            label_col=self.pred_column,
            prob_col=self.prob_column,
        )
        return DocumentDataset(df)
