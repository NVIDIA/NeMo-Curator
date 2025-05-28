"""Common Crawl download stage."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from ray_curator.data import DocumentBatch
from ray_curator.stages.base import ProcessingStage, StageType

logger = logging.getLogger(__name__)


class CommonCrawlDownloadStage(ProcessingStage[DocumentBatch]):
    """Download HTML content from Common Crawl based on URLs."""

    def __init__(
        self,
        url_column: str = "url",
        output_column: str = "html",
        max_workers: int = 10,
        timeout: int = 30,
        batch_size: int = 100,
    ):
        """Initialize the Common Crawl downloader.

        Args:
            url_column: Column containing Common Crawl URLs
            output_column: Column to store downloaded HTML
            max_workers: Maximum number of concurrent download threads
            timeout: Request timeout in seconds
            batch_size: Number of URLs to download in parallel
        """
        self.url_column = url_column
        self.output_column = output_column
        self.max_workers = max_workers
        self.timeout = timeout
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "common_crawl_download"

    @property
    def stage_type(self) -> StageType:
        return StageType.DOWNLOADER

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Download HTML content for URLs in the batch."""
        df = batch.to_pandas()

        # Check if URL column exists
        if self.url_column not in df.columns:
            logger.warning(f"URL column '{self.url_column}' not found in batch {batch.task_id}")
            return batch

        # Download HTML content
        logger.info(f"Downloading {len(df)} URLs from Common Crawl")
        html_content = self._download_batch(df[self.url_column].tolist())

        # Add HTML content to dataframe
        df[self.output_column] = html_content

        # Filter out failed downloads if needed
        success_mask = df[self.output_column].notna() & (df[self.output_column] != "")
        success_count = success_mask.sum()

        logger.info(f"Successfully downloaded {success_count}/{len(df)} documents")

        if success_count == 0:
            logger.warning(f"No successful downloads for batch {batch.task_id}")
            return None

        # Keep only successful downloads
        df = df[success_mask].copy()

        # Update additional columns
        additional_columns = list(batch.additional_columns)
        if self.output_column not in additional_columns:
            additional_columns.append(self.output_column)

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_downloaded",
            dataset_name=batch.dataset_name,
            data=df,
            metadata={
                **batch.metadata,
                "download_stats": {
                    "attempted": len(html_content),
                    "successful": success_count,
                    "failed": len(html_content) - success_count,
                },
                "downloader": self.name,
            },
            text_column=batch.text_column,
            id_column=batch.id_column,
            additional_columns=additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch

    def _download_batch(self, urls: list[str]) -> list[str | None]:
        """Download a batch of URLs in parallel."""
        results = [None] * len(urls)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_idx = {executor.submit(self._download_url, url): i for i, url in enumerate(urls)}

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Download failed for URL {urls[idx]}: {e}")
                    results[idx] = None

        return results

    def _download_url(self, url: str) -> str | None:
        """Download content from a single URL."""
        if pd.isna(url) or not url:
            return None

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return None
