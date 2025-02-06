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
    ):
        """
        Args:
            separator (str): The separator to join the documents on.
            text_field (str): The name of the column containing the text to join.
            segment_id_field (str): The name of the column containing the segment id.
        """
        self.separator = separator
        self.text_field = text_field
        self.segment_id_field = segment_id_field

    def _join_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Sort the segments so that they are in the correct order.
        df_sorted = df.sort_values(self.segment_id_field)
        # Group by the original document index (level 0) and join the segments using the separator.
        joined = df_sorted.groupby(level=0)[self.text_field].apply(
            lambda texts: self.separator.join(texts)
        )
        # Convert the result back to a DataFrame.
        return joined.to_frame(name=self.text_field)

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Joins the documents back into a single document.
        """
        # Construct meta information for the transformed dataframe.
        meta = dataset.df._meta.copy()
        if self.text_field not in meta.columns:
            meta[self.text_field] = pd.Series(dtype="object")

        # Apply the join operation partition-wise.
        dataset.df = dataset.df.map_partitions(self._join_partition, meta=meta)
        return dataset
