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
from typing import Any, Optional, Tuple

import fasttext
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.base import BaseModule
from nemo_curator.utils.distributed_utils import load_object_on_worker


class FastTextClassifier(BaseModule):
    """
    FastTextClassifier is a class designed to run any FastText model.
    It is the parent class for the DCLMFastTextClassifier.

    Attributes:
        model_path (str): The path to the .bin model file to use.
        model_identifier (str): Hugging Face identifier for the model, or any string to represent the model name.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "fasttext_quality_pred".
        prob_column (str): The column name where prediction probabilities will be stored. Defaults to "fasttext_quality_prob".
        high_quality_label (str): The string representation of the highest quality label assigned by the classifier.
            Defaults to "__label__hq".
    """

    def __init__(
        self,
        model_path: str,
        model_identifier: str,
        text_field: str = "text",
        pred_column: str = "fasttext_quality_pred",
        prob_column: str = "fasttext_quality_prob",
        high_quality_label: str = "__label__hq",
    ):
        super().__init__(input_backend="pandas")

        self.text_field = text_field
        self.pred_column = pred_column
        self.prob_column = prob_column
        self.high_quality_label = high_quality_label

        self.model_path = model_path
        self.model_identifier = model_identifier

    def _load_fasttext_model(self) -> Any:
        model = fasttext.load_model(self.model_path)
        return model

    def predict_text(self, text: str) -> Tuple[float, int]:
        model = load_object_on_worker(
            self.model_identifier, self._load_fasttext_model, {}
        )

        # predictions[0]: labels, predictions[1]: scores
        predictions = model.predict(text, k=len(model.get_labels()))

        # Return confidence of the highest quality label
        for i in range(len(predictions[0])):
            if predictions[0][i] == self.high_quality_label:
                # return confidence of high quality, actual label
                return predictions[1][i], predictions[0][0]

    def _predict_on_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        model = load_object_on_worker(
            self.model_identifier, self._load_fasttext_model, {}
        )
        results = df[self.text_field].apply(self.predict_text)

        df[self.prob_column] = results.apply(lambda x: x[0]).astype(np.float32)
        df[self.pred_column] = results.apply(lambda x: x[1]).astype(str)

        return df

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        meta = dataset.df._meta

        if hasattr(meta, "to_pandas"):
            meta = meta.to_pandas()

        meta[self.prob_column] = np.float32(0.0)
        meta[self.pred_column] = self.high_quality_label

        processed_df = dataset.df.to_backend("pandas").map_partitions(
            self._predict_on_partition, meta=meta
        )
        processed_df = processed_df.to_backend("cudf")

        return DocumentDataset(processed_df)


class DCLMFastTextClassifier(FastTextClassifier):
    """
    DCLMFastTextClassifier is a FastText model used for filtering in DataComp-LM to produce the DCLM-Baseline.
    It uses the FastText model from Hugging Face (https://huggingface.co/mlfoundations/fasttext-oh-eli5).

    Attributes:
        model_path (Optional[str]): The local path to the .bin model to use.
            If None, we read it from Hugging Face.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "dclm_fasttext_quality_pred".
        prob_column (str): The column name where prediction probabilities will be stored. Defaults to "dclm_fasttext_quality_prob".
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_identifier: Optional[str] = None,
        text_field: str = "text",
        pred_column: str = "dclm_fasttext_quality_pred",
        prob_column: str = "dclm_fasttext_quality_prob",
    ):
        if model_path is None:
            repo_id = "mlfoundations/fasttext-oh-eli5"
            filename = "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)

            if model_identifier is None:
                model_identifier = f"{repo_id}/{filename}"

        if model_path is not None and model_identifier is None:
            raise RuntimeError("model_identifier cannot be None")

        super().__init__(
            model_path=model_path,
            model_identifier=model_identifier,
            text_field=text_field,
            pred_column=pred_column,
            prob_column=prob_column,
            high_quality_label="__label__hq",
        )
