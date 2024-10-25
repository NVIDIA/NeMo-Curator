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

from typing import Optional

import dask
import fasttext
import numpy as np
import pandas as pd

from nemo_curator.filters.base import DocumentFilter, FilterMode
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import NoWorkerError, load_object_on_worker


class FastTextQualityFilter(DocumentFilter):
    def __init__(
        self,
        model_path=None,
        label="__label__hq",
        alpha=3,
        seed=42,
        text_field: str = "text",
        score_field: str = "fasttext_quality_score",
        filter_mode: FilterMode = FilterMode.SCORE_FILTER,
        removed_path: Optional[str] = None,
        invert: bool = False,
        save_score: bool = True,
    ):
        super().__init__(
            [float],
            text_fields=[text_field],
            score_fields=[score_field],
            filter_mode=filter_mode,
            removed_path=removed_path,
            invert=invert,
            save_score=save_score,
        )
        if model_path is None:
            raise ValueError(
                "Must provide a valid path to a FastText model "
                "to compute document scores with this filter"
            )
        self._model_path = model_path
        self._label = label
        self._alpha = alpha
        self._seed = np.random.seed(seed)
        self._name = "fasttext_quality_filter"

    @batched
    def score_document(self, df: pd.Series):
        model_attr = f"{self._name}_{self._model_path}"
        try:
            model = load_object_on_worker(model_attr, self._load_model, {})
        except NoWorkerError:
            return pd.Series(np.ones(len(df)), dtype=float)

        def _score_document(text):
            text = text.replace("\n", " ").replace("__label__", " ")
            pred = model.predict(text)
            document_score = pred[1][0]
            if pred[0][0] != self._label:
                document_score = 1 - document_score

            return document_score

        return df.apply(_score_document)

    @batched
    def keep_document(self, df: pd.Series):
        return np.random.pareto(self._alpha, size=len(df)) > 1 - df

    def _load_model(self):
        return fasttext.load_model(self._model_path)


class FastTextLangId(DocumentFilter):
    def __init__(
        self,
        model_path=None,
        min_langid_score=0.3,
        text_field: str = "text",
        score_field: str = "language_score",
        lang_field: str = "language",
        filter_mode: FilterMode = FilterMode.SCORE_FILTER,
        removed_path: Optional[str] = None,
        invert: bool = False,
        save_score: bool = True,
    ):
        super().__init__(
            [float],
            text_fields=[text_field],
            score_fields=[score_field, lang_field],
            filter_mode=filter_mode,
            removed_path=removed_path,
            invert=invert,
            save_score=save_score,
        )
        if model_path is None:
            raise ValueError(
                "Must provide a valid path to a FastText model "
                "to identify languages with this filter"
            )
        self.score_field = score_field
        self.lang_field = lang_field
        self._model_path = model_path
        self._lang_code = None
        self._cutoff = min_langid_score
        self._name = "lang_id"

    @batched
    def score_document(self, series: pd.Series):
        model_attr = f"{self._name}_{self._model_path}"
        try:
            model = load_object_on_worker(model_attr, self._load_model, {})
        except NoWorkerError:
            return pd.Series([[1.0, "N/A"] for _ in range(len(series))])

        processed_series = series.str.strip().str.replace("\n", " ")
        scores = []
        lang_codes = []
        for text in processed_series:
            label, score = model.predict(text, k=1)
            score = score[0]
            lang_code = label[0][-2:].upper()

            scores.append(score)
            lang_codes.append(lang_code)

        return pd.DataFrame(
            {self.score_field: scores, self.lang_field: lang_codes}, index=series.index
        )

    @batched
    def keep_document(self, score):
        return score[self.score_field] >= self._cutoff

    def _load_model(self):
        return fasttext.load_model(self._model_path)
