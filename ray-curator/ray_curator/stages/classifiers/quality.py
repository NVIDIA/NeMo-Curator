# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import cudf
import pandas as pd
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoTokenizer

from ray_curator.backends.base import WorkerMetadata

from .base import (
    DistributedDataClassifier,
    HFDeberta,
    StreamingDataClassifier,
    _run_classifier_helper,
)
from .utils import _get_suggest_memory_for_classifier

QUALITY_IDENTIFIER = "nvidia/quality-classifier-deberta"


@dataclass
class QualityModelConfig:
    model: str = "microsoft/deberta-v3-base"
    fc_dropout: float = 0.2
    max_len: int = 1024


class QualityModel(HFModel):
    def __init__(
        self,
        config: QualityModelConfig,
        autocast: bool = False,
        max_mem_gb: int | None = None,
    ):
        self.config = config
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()
        super().__init__(self.config.model, max_mem_gb=max_mem_gb)

    def load_model(self, device: str = "cuda") -> HFDeberta:
        model = HFDeberta.from_pretrained(QUALITY_IDENTIFIER)
        model.set_autocast(self.autocast)
        model = model.to(device)
        return model.eval()

    def load_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(QUALITY_IDENTIFIER)

    def load_config(self) -> AutoConfig:
        return AutoConfig.from_pretrained(QUALITY_IDENTIFIER)


class QualityClassifier(DistributedDataClassifier):
    """
    QualityClassifier is a specialized classifier designed for quality assessment tasks,
    utilizing the NemoCurator Quality Classifier DeBERTa model (https://huggingface.co/nvidia/quality-classifier-deberta).
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        filter_by (list[str], optional): The classes to filter the dataset by. If None, all classes will be included. Defaults to None.
        model_batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "quality_pred".
        prob_column (str): The column name where prediction probabilities will be stored. Defaults to "quality_prob".
        max_chars (int): The maximum number of characters in each document to consider for classification. Defaults to 6000.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                      it defaults to the available GPU memory minus 4 GB.

    """

    def __init__(  # noqa: PLR0913
        self,
        filter_by: list[str] | None = None,
        model_batch_size: int = 256,
        text_field: str = "text",
        pred_column: str = "quality_pred",
        prob_column: str = "quality_prob",
        max_chars: int = 6000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: int | None = None,
    ):
        config = AutoConfig.from_pretrained(QUALITY_IDENTIFIER)

        self.text_field = text_field
        self.prob_column = prob_column
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])
        self.out_dim = len(self.labels)
        self.max_mem_gb = max_mem_gb

        super().__init__(
            labels=self.labels,
            filter_by=filter_by,
            model_batch_size=model_batch_size,
            out_dim=self.out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    @property
    def name(self) -> str:
        return "quality_classifier"

    # TODO: Add setup_on_node function

    def setup(self, _: WorkerMetadata | None = None) -> None:
        # Load the Hugging Face model and processor from the cache.
        self.model = QualityModel(config=QualityModelConfig, autocast=self.autocast, max_mem_gb=self.max_mem_gb)

    def _run_classifier(self, df: pd.DataFrame | cudf.DataFrame) -> pd.DataFrame | cudf.DataFrame:
        print("Starting quality classifier inference", flush=True)
        return _run_classifier_helper(
            df=df,
            model=self.model,
            labels=self.labels,
            max_chars=self.max_chars,
            model_batch_size=self.model_batch_size,
            label_col=self.pred_column,
            text_field=self.text_field,
            prob_col=self.prob_column,
        )


class StreamingQualityClassifier(StreamingDataClassifier):
    def __init__(  # noqa: PLR0913
        self,
        filter_by: list[str] | None = None,
        model_batch_size: int = 256,
        text_field: str = "text",
        pred_column: str = "quality_pred",
        prob_column: str = "quality_prob",
        max_chars: int = 6000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: int | None = None,
        # TODO: Remove this once we decide on a permanent solution
        gpu_tokenizer: bool = True,
    ):
        config = AutoConfig.from_pretrained(QUALITY_IDENTIFIER)

        self.text_field = text_field
        self.prob_column = prob_column
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])
        self.out_dim = len(self.labels)
        self.max_mem_gb = max_mem_gb

        # TODO: Find a solution to remove this
        self.model = QualityModel(config=QualityModelConfig, autocast=autocast, max_mem_gb=self.max_mem_gb)

        super().__init__(
            labels=self.labels,
            filter_by=filter_by,
            model_batch_size=model_batch_size,
            out_dim=self.out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            gpu_tokenizer=gpu_tokenizer,
        )

    @property
    def name(self) -> str:
        return "streaming_quality_classifier"

    def setup(self, _: WorkerMetadata | None = None) -> None:
        # Load the Hugging Face model and processor from the cache.
        self.model = QualityModel(config=QualityModelConfig, autocast=self.autocast, max_mem_gb=self.max_mem_gb)
