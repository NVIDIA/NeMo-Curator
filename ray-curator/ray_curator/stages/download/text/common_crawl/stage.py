from dataclasses import dataclass
from typing import Literal

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.tasks import DocumentBatch, _EmptyTask

from .download import CommonCrawlWARCDownloader
from .html_extractor import CommonCrawlHTMLExtractor
from .html_extractors import HTMLExtractorAlgorithm
from .url_generation import MainCommonCrawlUrlStage, NewsCommonCrawlUrlStage
from .warc_reader import WarcReader


@dataclass
class CommonCrawl(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage for downloading and processing Common Crawl data.

    This pipeline:
    1. Generates WARC URLs (either from main or news crawls)
    2. Downloads WARC files
    3. Extracts content from WARC files
    4. Extracts text from HTML content
    """

    start_snapshot: str  # Format: YYYY-WW for main, YYYY-MM for news
    end_snapshot: str  # Format: YYYY-WW for main, YYYY-MM for news
    download_dir: str
    crawl_type: Literal["main", "news"] = "main"
    html_extraction: HTMLExtractorAlgorithm | str | None = None
    html_extraction_kwargs: dict | None = None
    stop_lists: dict[str, frozenset[str]] | None = None  # TODO: Find better name
    aws: bool = False
    verbose: bool = False
    limit: int | None = None

    def __post_init__(self):
        """Initialize the pipeline stages."""
        # URL generation stage
        if self.crawl_type == "main":
            url_stage = MainCommonCrawlUrlStage(
                start_snapshot_str=self.start_snapshot, end_snapshot_str=self.end_snapshot, limit=self.limit
            )
        else:
            url_stage = NewsCommonCrawlUrlStage(
                start_snapshot_str=self.start_snapshot, end_snapshot_str=self.end_snapshot, limit=self.limit
            )

        # Download stage
        download_stage = CommonCrawlWARCDownloader(download_dir=self.download_dir, aws=self.aws, verbose=self.verbose)

        # WARC processing stage
        warc_reader = WarcReader()

        # HTML extraction stage
        html_stage = CommonCrawlHTMLExtractor(
            algorithm=self.html_extraction,
            algorithm_kwargs=self.html_extraction_kwargs,
            stop_lists=self.stop_lists,
        )

        # Set up the pipeline
        self.stages = [
            url_stage,  # _EmptyTask -> FileGroupTask
            download_stage,  # FileGroupTask (cloud) -> FileGroupTask (local)
            warc_reader,  # FileGroupTask (local) -> DocumentBatch
            html_stage,  # DocumentBatch -> DocumentBatch
        ]

    @property
    def name(self) -> str:
        """Return the name of this stage."""
        return f"common_crawl_{self.crawl_type}_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return f"Common Crawl {self.crawl_type} pipeline: {self.start_snapshot} to {self.end_snapshot}"
