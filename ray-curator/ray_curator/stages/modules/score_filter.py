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

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.tasks import DocumentBatch


@dataclass
class Score(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for adding metadata to records based on statistics about the text.
    It accepts an arbitrary scoring function that accepts a text field and returns a score.
    It also accepts a DocumentFilter object, in which case the score_fn will be the score_document method of the DocumentFilter.

    Unlike ScoreFilter, it does not filter based on the computed score.
    It only adds metadata to the record.

    Args:
        score_fn (Callable | DocumentFilter): The score function or the DocumentFilter object. If it is a DocumentFilter object, the score_fn will be the score_document method of the DocumentFilter.
        score_field (str): The field the score will be stored in.
        text_field (str): The field the documents will be read from.
        processing_batch_size (int): The number of tasks to process in a batch.

    """

    score_fn: Callable[[str], float | str] | DocumentFilter
    score_field: str
    text_field: str = "text"
    # TODO: Remove this once we have .with(batch_size=...)
    processing_batch_size: int = 1

    @property
    def name(self) -> str:
        return self.score_fn.name if isinstance(self.score_fn, DocumentFilter) else "score_fn"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.score_field]

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return self.processing_batch_size

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if isinstance(self.score_fn, DocumentFilter) and hasattr(self.score_fn, "model_check_or_download"):
            self.score_fn.model_check_or_download()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if isinstance(self.score_fn, DocumentFilter):
            if hasattr(self.score_fn, "load_model"):
                self.score_fn.load_model()
            elif hasattr(self.score_fn, "load_tokenizer"):
                self.score_fn.load_tokenizer()

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Scores all records in multiple dataset batches

        Args:
            tasks (list[DocumentBatch]): The dataset batches to apply the module to

        Returns:
            list[DocumentBatch]: The dataset batches with the score applied

        """

        if self.processing_batch_size == 1:
            return [self.process(tasks[0])]

        # Convert and collect all Pandas DataFrames
        dfs = [task.to_pandas() for task in tasks]
        lengths = [len(df) for df in dfs]
        task_ids = [task.task_id for task in tasks]
        dataset_names = [task.dataset_name for task in tasks]

        # Combine and process
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_batch = DocumentBatch(data=combined_df, task_id="batch_list", dataset_name="batch_list")
        processed_batch = self.process(combined_batch)
        processed_df = processed_batch.to_pandas()

        # Use original lengths to rebuild the DocumentBatch objects
        result_batches = []
        offset = 0
        for length, task_id, dataset_name in zip(lengths, task_ids, dataset_names, strict=True):
            chunk = processed_df.iloc[offset : offset + length].reset_index(drop=True)
            result_batches.append(
                DocumentBatch(
                    task_id=f"{task_id}_{self.name}",
                    dataset_name=dataset_name,
                    data=chunk,
                )
            )
            offset += length

        return result_batches

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the scoring to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the new score

        """

        if isinstance(self.score_fn, DocumentFilter):
            self.score_fn = self.score_fn.score_document

        df = batch.to_pandas()
        df[self.score_field] = df[self.text_field].apply(self.score_fn)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
        )


@dataclass
class Filter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for filtering records based on a metadata field.
    It accepts an arbitrary filter function that accepts a metadata field and returns True if the field should be kept.
    It also accepts a DocumentFilter object, in which case the filter_fn will be the keep_document method of the DocumentFilter.
    Unlike ScoreFilter, it does not compute the metadata based on a document.
    It only filters using existing metadata.

    Args:
        filter_fn (Callable | DocumentFilter): A function that returns True if the document is to be kept or a DocumentFilter object,
            in which case the filter_fn will be the keep_document method of the DocumentFilter.
        filter_field (str): The field(s) to be passed into the filter function.
        invert (bool): Whether to invert the filter condition.
        processing_batch_size (int): The number of tasks to process in a batch.

    """

    filter_fn: Callable | DocumentFilter
    filter_field: str
    invert: bool = False
    # TODO: Remove this once we have .with(batch_size=...)
    processing_batch_size: int = 1

    @property
    def name(self) -> str:
        return self.filter_fn.name if isinstance(self.filter_fn, DocumentFilter) else "filter_fn"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filter_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return self.processing_batch_size

    def compute_filter_mask(self, df: pd.DataFrame) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        if isinstance(self.filter_fn, DocumentFilter):
            self.filter_fn = self.filter_fn.keep_document

        bool_mask = df[self.filter_field].apply(self.filter_fn)

        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Filters records in multiple dataset batches

        Args:
            tasks (list[DocumentBatch]): The dataset batches to apply the module to

        Returns:
            list[DocumentBatch]: The dataset batches with the filter applied

        """

        if self.processing_batch_size == 1:
            return [self.process(tasks[0])]

        dfs = []
        for i, task in enumerate(tasks):
            df = task.to_pandas().copy()
            df["__original_batch_id__"] = i
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        task_ids = [task.task_id for task in tasks]
        dataset_names = [task.dataset_name for task in tasks]
        batch_id_str = "_".join(task_ids)
        batch_name_str = "_".join(dataset_names)
        combined_batch = DocumentBatch(data=combined_df, task_id=batch_id_str, dataset_name=batch_name_str)

        processed_batch = self.process(combined_batch)

        if processed_batch is None:
            return []

        filtered_df = processed_batch.to_pandas()

        if "__original_batch_id__" not in filtered_df.columns:
            msg = "Expected '__original_batch_id__' column is missing after processing."
            raise ValueError(msg)

        # Reconstruct batches based on the temp column
        result_batches = []
        for i, group_df in filtered_df.groupby("__original_batch_id__"):
            task = tasks[i]
            cleaned_df = group_df.drop(columns="__original_batch_id__").reset_index(drop=True)
            result_batches.append(
                DocumentBatch(
                    task_id=f"{task.task_id}_{self.name}",
                    dataset_name=task.dataset_name,
                    data=cleaned_df,
                )
            )

        return result_batches

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the filtering to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with entries removed according to the filter

        """
        df = batch.to_pandas()
        bool_mask = self.compute_filter_mask(df)
        result_df = df[bool_mask]

        if len(result_df) == 0:
            print(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )


@dataclass
class ScoreFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for applying a filter to all documents in a dataset.
    It accepts an arbitrary DocumentFilter and first computes the score for a document.
    Then, determines whether to keep the document based on the criteria in the DocumentFilter.

    The filter can be applied to any field in the dataset, and the score can be logged for later.
    Also, the filter can be inverted such that "rejected" documents are kept.

    Args:
        filter_obj (DocumentFilter): The score function that takes in a document string and outputs a score for the document.
        text_field (str): The field the documents will be read from.
        score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
        invert (bool): If True, will keep all documents that are normally discarded.
        processing_batch_size (int): The number of tasks to process in a batch.

    """

    filter_obj: DocumentFilter
    text_field: str = "text"
    score_field: str | None = None
    invert: bool = False
    # TODO: Remove this once we have .with(batch_size=...)
    processing_batch_size: int = 1

    @property
    def name(self) -> str:
        return self.filter_obj.name

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.score_field] if self.score_field else []

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return self.processing_batch_size

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if isinstance(self.filter_obj, DocumentFilter) and hasattr(self.filter_obj, "model_check_or_download"):
            self.filter_obj.model_check_or_download()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if isinstance(self.filter_obj, DocumentFilter):
            if hasattr(self.filter_obj, "load_model"):
                self.filter_obj.load_model()
            elif hasattr(self.filter_obj, "load_tokenizer"):
                self.filter_obj.load_tokenizer()

    def compute_filter_mask(self, df: pd.DataFrame) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        scores = df[self.text_field].apply(self.filter_obj.score_document)

        if self.score_field is not None:
            df[self.score_field] = scores

        bool_mask = scores.apply(self.filter_obj.keep_document)

        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Scores and filters all records in multiple dataset batches

        Args:
            tasks (list[DocumentBatch]): The dataset batches to apply the module to

        Returns:
            list[DocumentBatch]: The dataset batches with the score and filter applied

        """

        if self.processing_batch_size == 1:
            return [self.process(tasks[0])]

        dfs = []
        for i, task in enumerate(tasks):
            df = task.to_pandas().copy()
            df["__original_batch_id__"] = i
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        task_ids = [task.task_id for task in tasks]
        dataset_names = [task.dataset_name for task in tasks]
        batch_id_str = "_".join(task_ids)
        batch_name_str = "_".join(dataset_names)
        combined_batch = DocumentBatch(data=combined_df, task_id=batch_id_str, dataset_name=batch_name_str)

        processed_batch = self.process(combined_batch)

        if processed_batch is None:
            return []

        filtered_df = processed_batch.to_pandas()

        if "__original_batch_id__" not in filtered_df.columns:
            msg = "Expected '__original_batch_id__' column is missing after processing."
            raise ValueError(msg)

        # Reconstruct batches based on the temp column
        result_batches = []
        for i, group_df in filtered_df.groupby("__original_batch_id__"):
            task = tasks[i]
            cleaned_df = group_df.drop(columns="__original_batch_id__").reset_index(drop=True)
            result_batches.append(
                DocumentBatch(
                    task_id=f"{task.task_id}_{self.name}",
                    dataset_name=task.dataset_name,
                    data=cleaned_df,
                )
            )

        return result_batches

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Scores and filters all records in the dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the score and filter applied

        """
        df = batch.to_pandas()
        bool_mask = self.compute_filter_mask(df)
        result_df = df[bool_mask]

        if len(result_df) == 0:
            print(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )
