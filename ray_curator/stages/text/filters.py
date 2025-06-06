"""Text-specific filter stages."""

import logging

import pandas as pd

from ray_curator.data import DocumentBatch
from ray_curator.stages.base import ProcessingStage, StageType

logger = logging.getLogger(__name__)


class TextLengthFilterStage(ProcessingStage[DocumentBatch]):
    """Filter documents based on text length."""

    def __init__(self, min_length: int = 0, max_length: int | None = None):
        """Initialize the text length filter.
        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters (None for no limit)
        """
        self.min_length = min_length
        self.max_length = max_length

    @property
    def name(self) -> str:
        return "text_length_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Filter documents by text length."""
        df = batch.to_pandas()
        initial_count = len(df)

        # Get text column
        text_col = batch.text_column

        # Apply length filter
        mask = df[text_col].str.len() >= self.min_length
        if self.max_length is not None:
            mask &= df[text_col].str.len() <= self.max_length

        filtered_df = df[mask].copy()

        filter_stats = {
            "total": initial_count,
            "passed": len(filtered_df),
            "filtered": initial_count - len(filtered_df),
            "min_length": self.min_length,
            "max_length": self.max_length,
        }

        logger.info(f"Text length filter stats: {filter_stats}")

        if len(filtered_df) == 0:
            logger.warning(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_length_filtered",
            dataset_name=batch.dataset_name,
            data=filtered_df,
            metadata={**batch.metadata, "filter_stats": filter_stats, "filter": self.name},
            text_column=batch.text_column,
            id_column=batch.id_column,
            additional_columns=batch.additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch


class TextLanguageFilterStage(ProcessingStage[DocumentBatch]):
    """Filter documents based on language."""

    def __init__(self, languages: list[str], language_column: str = "language"):
        """Initialize the language filter.
        Args:
            languages: List of accepted language codes (e.g., ["en", "es"])
            language_column: Column containing language information
        """
        self.languages = [lang.lower() for lang in languages]
        self.language_column = language_column

    @property
    def name(self) -> str:
        return "text_language_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Filter documents by language."""
        df = batch.to_pandas()
        initial_count = len(df)

        # Check if language column exists
        if self.language_column not in df.columns:
            logger.warning(f"Language column '{self.language_column}' not found in batch {batch.task_id}")
            return batch

        # Apply language filter
        mask = df[self.language_column].str.lower().isin(self.languages)
        filtered_df = df[mask].copy()

        filter_stats = {
            "total": initial_count,
            "passed": len(filtered_df),
            "filtered": initial_count - len(filtered_df),
            "accepted_languages": self.languages,
        }

        logger.info(f"Language filter stats: {filter_stats}")

        if len(filtered_df) == 0:
            logger.warning(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_language_filtered",
            dataset_name=batch.dataset_name,
            data=filtered_df,
            metadata={**batch.metadata, "filter_stats": filter_stats, "filter": self.name},
            text_column=batch.text_column,
            id_column=batch.id_column,
            additional_columns=batch.additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch


class TextContentFilterStage(ProcessingStage[DocumentBatch]):
    """Filter documents based on content patterns."""

    def __init__(self, patterns: list[str], mode: str = "contains", case_sensitive: bool = False):
        """Initialize the content filter.
        Args:
            patterns: List of patterns to search for
            mode: "contains" (keep if contains any pattern) or
                  "excludes" (remove if contains any pattern)
            case_sensitive: Whether pattern matching is case sensitive
        """
        self.patterns = patterns
        self.mode = mode
        self.case_sensitive = case_sensitive

        if mode not in ["contains", "excludes"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'contains' or 'excludes'")

    @property
    def name(self) -> str:
        return "text_content_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Filter documents by content patterns."""
        df = batch.to_pandas()
        initial_count = len(df)

        # Get text column
        text_col = batch.text_column

        # Create pattern matching mask
        if self.case_sensitive:
            pattern_mask = df[text_col].apply(lambda x: any(pattern in x for pattern in self.patterns))
        else:
            lower_patterns = [p.lower() for p in self.patterns]
            pattern_mask = df[text_col].str.lower().apply(lambda x: any(pattern in x for pattern in lower_patterns))

        # Apply based on mode
        if self.mode == "contains":
            mask = pattern_mask
        else:  # excludes
            mask = ~pattern_mask

        filtered_df = df[mask].copy()

        filter_stats = {
            "total": initial_count,
            "passed": len(filtered_df),
            "filtered": initial_count - len(filtered_df),
            "mode": self.mode,
            "patterns": self.patterns[:5],  # Show first 5 patterns
        }

        logger.info(f"Content filter stats: {filter_stats}")

        if len(filtered_df) == 0:
            logger.warning(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_content_filtered",
            dataset_name=batch.dataset_name,
            data=filtered_df,
            metadata={**batch.metadata, "filter_stats": filter_stats, "filter": self.name},
            text_column=batch.text_column,
            id_column=batch.id_column,
            additional_columns=batch.additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch


class TextMetadataFilterStage(ProcessingStage[DocumentBatch]):
    """Filter documents based on metadata values."""

    def __init__(self, filters: dict):
        """Initialize the metadata filter.
        Args:
            filters: Dictionary of column_name -> accepted_values
                    e.g., {"source": ["wikipedia", "books"], "quality": ["high"]}
        """
        self.filters = filters

    @property
    def name(self) -> str:
        return "text_metadata_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Filter documents by metadata."""
        df = batch.to_pandas()
        initial_count = len(df)

        # Apply all filters
        mask = pd.Series([True] * len(df), index=df.index)

        for column, accepted_values in self.filters.items():
            if column in df.columns:
                if isinstance(accepted_values, list):
                    mask &= df[column].isin(accepted_values)
                else:
                    mask &= df[column] == accepted_values
            else:
                logger.warning(f"Column '{column}' not found in batch {batch.task_id}")

        filtered_df = df[mask].copy()

        filter_stats = {
            "total": initial_count,
            "passed": len(filtered_df),
            "filtered": initial_count - len(filtered_df),
            "filters": self.filters,
        }

        logger.info(f"Metadata filter stats: {filter_stats}")

        if len(filtered_df) == 0:
            logger.warning(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_metadata_filtered",
            dataset_name=batch.dataset_name,
            data=filtered_df,
            metadata={**batch.metadata, "filter_stats": filter_stats, "filter": self.name},
            text_column=batch.text_column,
            id_column=batch.id_column,
            additional_columns=batch.additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch
