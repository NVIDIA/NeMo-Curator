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
from typing import List

import pandas as pd

from nemo_curator.datasets import DocumentDataset


class DocumentSplitter:
    """
    Splits documents into segments based on a separator.
    Each segment is a new document with an additional column
    indicating the segment id.

    To restore the original document, ensure that each document
    has a unique id prior to splitting.
    """

    def __init__(
        self,
        separator: str,
        text_field: str = "text",
        segment_id_field: str = "segment_id",
    ):
        """
        Args:
            separator (str): The separator to split the documents on.
            text_field (str): The name of the column containing the text to split.
            segment_id_field (str): The name of the column to add to indicate the segment id.
        """
        self.separator = separator
        self.text_field = text_field
        self.segment_id_field = segment_id_field

    def _split_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy to avoid modifying the original dataframe in place.
        df = df.copy()
        # Split the text field into segments using the separator.
        df["split_text"] = df[self.text_field].str.split(self.separator)
        # Explode the list so that each segment becomes a separate row.
        df = df.explode("split_text")
        # For each original document (grouped by the original index), assign a segment id.
        df[self.segment_id_field] = df.groupby(level=0).cumcount()
        # Replace the original text field with the split segment.
        df[self.text_field] = df["split_text"]
        # Drop the temporary column.
        df = df.drop(columns="split_text")
        return df

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Splits the documents into segments based on the separator and
        adds a column indicating the segment id.
        """

        # Construct meta information for the transformed dataframe.
        meta = dataset.df._meta.copy()
        if self.segment_id_field not in meta.columns:
            meta[self.segment_id_field] = pd.Series(dtype="int64")

        # Apply the partition-wise splitting transformation using Dask's map_partitions.
        dataset.df = dataset.df.map_partitions(self._split_partition, meta=meta)
        return dataset


class DocumentJoiner:
    """
    Joins documents that have a common id back into a single document.
    The order of the documents is dictated by an additional segment_id column.

    The joined documents are joined by a separator.
    """

    def __init__(
        self,
        separator: str,
        text_field: str = "text",
        segment_id_field: str = "segment_id",
        document_id_field: str = "id",
        drop_segment_id_field: bool = True,
    ):
        """
        Args:
            separator (str): The separator to join the documents on.
            text_field (str): The name of the column containing the text to join.
            segment_id_field (str): The name of the column containing the segment id.
            document_id_field (str): The name of the column containing the document id.
            drop_segment_id_field (bool): Whether to drop the segment_id_field after joining.
        """
        self.separator = separator
        self.text_field = text_field
        self.segment_id_field = segment_id_field
        self.document_id_field = document_id_field
        self.drop_segment_id_field = drop_segment_id_field

    def _join_partition(
        self, df: pd.DataFrame, expected_cols: List[str]
    ) -> pd.DataFrame:
        if df.empty:
            return df
        # Sort the segments by the segment_id_field to maintain proper order before aggregating.
        df_sorted = df.sort_values(self.segment_id_field)
        # Build aggregation functions to preserve all original columns:
        # - For self.text_field, join all segments using the separator.
        # - For all other columns (except self.document_id_field, which is our grouping key), take the first occurrence.
        agg_funcs = {}
        for col in df_sorted.columns:
            if col == self.text_field:
                agg_funcs[col] = lambda texts: self.separator.join(texts.astype(str))
            elif col != self.document_id_field:
                agg_funcs[col] = "first"
        # Group by document_id_field while keeping the key as a column.
        joined = df_sorted.groupby(self.document_id_field, as_index=False).agg(
            agg_funcs
        )

        if self.drop_segment_id_field:
            joined = joined.drop(columns=self.segment_id_field)
        # Reorder the columns to match the expected metadata order.
        joined = joined[expected_cols]
        return joined

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Joins the documents back into a single document while preserving all the original fields.
        """
        # Construct meta information for the transformed dataframe.
        meta = dataset.df._meta.copy()
        if self.text_field not in meta.columns:
            meta[self.text_field] = pd.Series(dtype="object")
        # If dropping the segment id field, remove it from the metadata to prevent mismatches.
        if self.drop_segment_id_field:
            meta = meta.drop(columns=self.segment_id_field)
        expected_cols = list(meta.columns)
        # Apply the join operation partition-wise.
        dataset.df = dataset.df.map_partitions(
            self._join_partition, expected_cols=expected_cols, meta=meta
        )
        return dataset
