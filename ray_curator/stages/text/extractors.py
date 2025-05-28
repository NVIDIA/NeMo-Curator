"""Text extraction stages."""

import logging
from typing import Any

import pandas as pd
import trafilatura
from bs4 import BeautifulSoup

from ray_curator.data import DocumentBatch
from ray_curator.stages.base import ProcessingStage, StageType

logger = logging.getLogger(__name__)


class HtmlTextExtractorStage(ProcessingStage[DocumentBatch]):
    """Extract text content from HTML documents."""

    def __init__(
        self,
        method: str = "trafilatura",
        extract_metadata: bool = True,
        html_column: str = "html",
        output_column: str = "extracted_text",
    ):
        """Initialize the HTML text extractor.

        Args:
            method: Extraction method ("trafilatura" or "beautifulsoup")
            extract_metadata: Whether to extract metadata from HTML
            html_column: Column containing HTML content
            output_column: Column to store extracted text
        """
        self.method = method
        self.extract_metadata = extract_metadata
        self.html_column = html_column
        self.output_column = output_column

        if method not in ["trafilatura", "beautifulsoup"]:
            raise ValueError(f"Unknown extraction method: {method}")

    @property
    def name(self) -> str:
        return "html_text_extractor"

    @property
    def stage_type(self) -> StageType:
        return StageType.EXTRACTOR

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Extract text from HTML documents."""
        df = batch.to_pandas()

        # Check if HTML column exists
        if self.html_column not in df.columns:
            logger.warning(f"HTML column '{self.html_column}' not found in batch {batch.task_id}")
            return batch

        # Extract text from each document
        if self.method == "trafilatura":
            extracted_data = df[self.html_column].apply(self._extract_with_trafilatura)
        else:
            extracted_data = df[self.html_column].apply(self._extract_with_beautifulsoup)

        # Handle extraction results
        if self.extract_metadata and self.method == "trafilatura":
            # Trafilatura returns dict with text and metadata
            df[self.output_column] = extracted_data.apply(lambda x: x.get("text", "") if isinstance(x, dict) else x)
            # Add metadata columns
            metadata_df = pd.json_normalize(
                extracted_data.apply(lambda x: x.get("metadata", {}) if isinstance(x, dict) else {})
            )
            for col in metadata_df.columns:
                df[f"html_{col}"] = metadata_df[col]
        else:
            # Just text
            df[self.output_column] = extracted_data

        # Update text column if we're replacing the main text
        new_text_column = batch.text_column
        if self.output_column != batch.text_column:
            # Add to additional columns
            additional_columns = list(batch.additional_columns)
            if self.output_column not in additional_columns:
                additional_columns.append(self.output_column)
        else:
            new_text_column = self.output_column
            additional_columns = batch.additional_columns

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_html_extracted",
            dataset_name=batch.dataset_name,
            data=df,
            metadata={**batch.metadata, "html_extraction_method": self.method, "extractor": self.name},
            text_column=new_text_column,
            id_column=batch.id_column,
            additional_columns=additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch

    def _extract_with_trafilatura(self, html: str) -> dict[str, Any]:
        """Extract text using trafilatura."""
        if pd.isna(html) or not html:
            return {"text": "", "metadata": {}}

        try:
            # Extract text
            text = trafilatura.extract(html, include_comments=False, include_tables=True, deduplicate=True)

            if self.extract_metadata:
                # Extract metadata
                metadata = trafilatura.extract_metadata(html)
                return {"text": text or "", "metadata": metadata.__dict__ if metadata else {}}
            else:
                return {"text": text or "", "metadata": {}}

        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            return {"text": "", "metadata": {}}

    def _extract_with_beautifulsoup(self, html: str) -> str:
        """Extract text using BeautifulSoup."""
        if pd.isna(html) or not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed: {e}")
            return ""
