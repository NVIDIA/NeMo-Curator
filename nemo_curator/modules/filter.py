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

from typing import Callable, List, Optional, Union

import pandas as pd
from dask.array import logical_and
from dask.dataframe.extensions import make_array_nonempty
from dask.typing import no_default

from nemo_curator.datasets import DocumentDataset
from nemo_curator.datasets.parallel_dataset import ParallelDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.modules.base import BaseModule
from nemo_curator.utils.module_utils import REASON_LABEL_KEY, SKIP_LABEL_KEY, is_batched

# Override so that pd.NA is not passed during the metadata inference
make_array_nonempty.register(
    pd.StringDtype,
    lambda x: pd.array(["a", "b"], dtype=x),
)


class Score(BaseModule):
    """
    The module responsible for adding metadata to records based on statistics about the text.
    It accepts an arbitrary scoring function that accepts a text field and returns a score.

    Unlike ScoreFilter, it does not filter based on the computed score.
    It only adds metadata to the record.
    """

    def __init__(
        self,
        score_fn: Callable,
        score_field: str,
        text_field: str = "text",
        score_type: Union[type, str] = None,
    ):
        """
        Constructs a Score module.

        Args:
          score_fn (Callable): The score function that takes in a document string and outputs a score for the document.
          score_field (str): The field the score will be stored in.
          text_field (str): The field the documents will be read from.
          score_type (Union[type, str]): The datatype of the score that will be made for each document.
        """
        super().__init__(input_backend="pandas")
        self.score_fn = score_fn
        self.score_field = score_field
        self.text_field = text_field
        self.score_type = score_type

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Applies the scoring to a dataset

        Args:
            dataset (DocumentDataset): The dataset to apply the module to

        Returns:
            DocumentDataset: A dataset with the new score
        """
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        if is_batched(self.score_fn):
            dataset.df[self.score_field] = dataset.df[self.text_field].map_partitions(
                self.score_fn, meta=meta
            )
        else:
            dataset.df[self.score_field] = dataset.df[self.text_field].apply(
                self.score_fn, meta=meta
            )

        return dataset


class Filter(BaseModule):
    """
    The module responsible for filtering records based on a metadata field.
    It accepts an arbitrary filter function that accepts a metadata field and returns True if the field should be kept.

    Unlike ScoreFilter, it does not compute the metadata based on a document.
    It only filters using existing metadata.
    """

    def __init__(self, filter_fn: Callable, filter_field: str, invert: bool = False):
        """
        Constructs a Filter module

        Args:
          filter_fn (Callable): A function that returns True if the document is to be kept.
          filter_field (str): The field(s) to be passed into the filter function.
          invert (bool): Whether to invert the filter condition.
        """
        super().__init__(input_backend="pandas")
        self.filter_fn = filter_fn
        self.filter_field = filter_field
        self.invert = invert

    def compute_filter_mask(self, dataset: DocumentDataset):
        """Compute the bool mask to filter the dataset.

        Args:
            dataset (DocumentDataset): The dataset to compute filter mask on.

        Returns:
            Series or DataFrame: A mask corresponding to each data instance indicating whether it will be retained.
        """
        if is_batched(self.filter_fn):
            bool_mask = dataset.df[self.filter_field].map_partitions(
                self.filter_fn, meta=(None, bool)
            )
        else:
            bool_mask = dataset.df[self.filter_field].apply(
                self.filter_fn, meta=(None, bool)
            )

        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Applies the filtering to a dataset

        Args:
            dataset (DocumentDataset): The dataset to apply the module to

        Returns:
            DocumentDataset: A dataset with entries removed according to the filter
        """
        bool_mask = self.compute_filter_mask(dataset)
        return DocumentDataset(dataset.df[bool_mask])


class ScoreFilter(BaseModule):
    """
    The module responsible for applying a filter to all documents in a DocumentDataset.
    It accepts an arbitrary DocumentFilter and first computes the score for a document.
    Then, determines whether to keep the document based on the criteria in the DocumentFilter.

    The filter can be applied to any field in the dataset, and the score can be logged for later.
    Also, the filter can be inverted such that "rejected" documents are kept.
    """

    def __init__(
        self,
        filter_obj: DocumentFilter,
        text_field: str = "text",
        score_field: Optional[str] = None,
        score_type: Union[type, str] = None,
        invert: bool = False,
        add_skip_label_only: bool = False,
    ):
        """
        Constructs a ScoreFilter module.

        Args:
          filter_obj (DocumentFilter): The score function that takes in a document string and outputs a score for the document.
          text_field (str): The field the documents will be read from.
          score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
          score_type (Union[type, str]): The datatype of the score that will be made for each document.
          invert (bool): If True, will keep all documents that are normally discarded.
        """
        super().__init__(input_backend=filter_obj.backend)
        self.filter_obj = filter_obj
        self.text_field = text_field
        self.score_field = score_field
        self.score_type = score_type
        self.invert = invert
        self.add_skip_label_only = add_skip_label_only

    def compute_filter_mask(self, dataset: DocumentDataset):
        """Compute the bool mask to filter the dataset.

        Args:
            dataset (DocumentDataset): The dataset to compute filter mask on.

        Returns:
            Series or DataFrame: A mask corresponding to each data instance indicating whether it will be retained.
        """
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        if self.add_skip_label_only:
            if SKIP_LABEL_KEY not in dataset.df.columns:
                dataset.df[SKIP_LABEL_KEY] = 0
            if REASON_LABEL_KEY not in dataset.df.columns:
                dataset.df[REASON_LABEL_KEY] = None

            # although the full dataset is passed, we don't need to compute score on full data
            # only data that's still remaining needs to be processed
            kept_mask = dataset.df._skipme == 0
            df = dataset.df[kept_mask].copy()
        else:
            df = dataset.df

        if is_batched(self.filter_obj.score_document):
            scores = df[self.text_field].map_partitions(
                self.filter_obj.score_document, meta=meta
            )
        else:
            scores = df[self.text_field].apply(
                self.filter_obj.score_document, meta=meta
            )

        if self.score_field is not None and self.add_skip_label_only:
            dataset.df[self.score_field] = None

            def update_score(partition, mask_partition, score_partition):
                partition.loc[mask_partition, self.score_field] = score_partition
                return partition

            dataset.df = dataset.df.map_partitions(
                update_score, kept_mask, scores, meta=dataset.df
            )
        elif self.score_field is not None:
            dataset.df[self.score_field] = scores

        if is_batched(self.filter_obj.keep_document):
            bool_mask = scores.map_partitions(
                self.filter_obj.keep_document, meta=(None, bool)
            )
        else:
            bool_mask = scores.apply(self.filter_obj.keep_document, meta=(None, bool))
        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Scores and filters all records in the dataset

        Args:
            dataset (DocumentDataset): The dataset to apply the module to

        Returns:
            DocumentDataset: A dataset with the score and filter applied
        """
        bool_mask = self.compute_filter_mask(dataset)

        def update_skipme(partition, kept_mask_partition, score_bool_mask_partition):
            partition.loc[kept_mask_partition, SKIP_LABEL_KEY] = [
                1 if skip else 0 for skip in ~score_bool_mask_partition
            ]
            return partition

        def update_reason(partition, kept_mask_partition, reason):
            # filtering reason needs to be updated for the following entries
            # 1. the entry was kept before
            # 2. the entry was thrown out by this filter
            new_skip = [
                True if skip == 1 else False for skip in partition[SKIP_LABEL_KEY]
            ]
            new_mask = logical_and(kept_mask_partition.values, new_skip)
            partition.loc[new_mask, REASON_LABEL_KEY] = reason
            return partition

        if self.add_skip_label_only:
            kept_mask = dataset.df._skipme == 0
            dataset.df = dataset.df.map_partitions(
                update_skipme,
                kept_mask,
                ~bool_mask if self.invert else bool_mask,
                meta=dataset.df,
            )
            dataset.df = dataset.df.map_partitions(
                update_reason,
                kept_mask,
                self.filter_obj.__class__.__name__,
                meta=dataset.df,
            )
            return DocumentDataset(dataset.df)
        else:
            return DocumentDataset(dataset.df[bool_mask])


class ParallelScoreFilter(BaseModule):
    def __init__(
        self,
        src_filter_obj,
        tgt_filter_obj,
        src_field="src",
        tgt_field="tgt",
        src_score=None,
        tgt_score=None,
        score_type=None,
        invert=False,
        add_skip_label_only: bool = False,
    ):
        """A filter object wrapper class for applying *monolingual* filter objects on bitext.
        If either side of the bitext is discarded, the whole bitext pair is discarded.
        If you want to apply a *bitext* filter that takes both the source and target as input, checkout `BitextFilter` class.

        Note that the goal of this wrapper class is to group the same/similar filters on bitext thus making the logic clearer,
        which is why we force the `score_type` and `invert` to be the same among source/target filters.
        If you need the extra flexibility, you should fall back to applying two filters one after the other.

        Args:
            src_filter_obj (_type_): The score function that takes in a document string and outputs a score for the source document.
            tgt_filter_obj (_type_): The score function that takes in a document string and outputs a score for the target document.
            src_field (str, optional): The field the source documents will be read from. Defaults to "src".
            tgt_field (str, optional): The field the target documents will be read from. Defaults to "tgt".
            src_score (str, optional): The field to which the source scores will be written. If None, scores will be immediately discarded after use. Defaults to None.
            tgt_score (str, optional): The field to which the target scores will be written. If None, scores will be immediately discarded after use. Defaults to None.
            score_type (Optional[str]): The datatype of the score that will be made for each document. Defaults to None.
            invert (bool, optional): If True, will keep all documents that are normally discarded. Defaults to False.
        """
        super().__init__(input_backend=src_filter_obj.backend)
        self.source_score_filter = ScoreFilter(
            src_filter_obj,
            src_field,
            src_score,
            score_type,
            invert,
            add_skip_label_only,
        )
        self.target_score_filter = ScoreFilter(
            tgt_filter_obj,
            tgt_field,
            tgt_score,
            score_type,
            invert,
            add_skip_label_only,
        )
        self.add_skip_label_only = add_skip_label_only

    def call(self, dataset: ParallelDataset) -> ParallelDataset:
        # remove lines together if one of them is filtered
        ds1 = self.source_score_filter(dataset)
        ds2 = self.target_score_filter(ds1)
        return ParallelDataset(ds2.df)
