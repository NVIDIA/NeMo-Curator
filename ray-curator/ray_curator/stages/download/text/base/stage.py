from dataclasses import dataclass

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.tasks import DocumentBatch, _EmptyTask

from .download import DocumentDownloader, DocumentDownloadStage
from .extract import DocumentExtractor, DocumentExtractStage
from .iterator import DocumentIterateStage, DocumentIterator
from .url_generation import URLGenerationStage, URLGenerator


@dataclass
class DocumentDownloadExtractStage(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage that combines URL generation, download, iterate, and extract stages.

    This supports the full 4-step pipeline pattern like Common Crawl:
    1. Generate URLs from minimal input
    2. Download files from URLs
    3. Iterate through files to extract raw records
    4. Extract structured content from raw records

    """

    url_generator: URLGenerator
    downloader: DocumentDownloader
    iterator: DocumentIterator
    extractor: DocumentExtractor | None = None
    url_limit: int | None = None
    record_limit: int | None = None
    add_filename_column: bool | str = True

    def __post_init__(self):
        """Initialize the constituent stages."""
        # URL generation stage
        url_stage = URLGenerationStage(
            url_generator=self.url_generator,
            limit=self.url_limit,
        )

        # Download stage
        download_stage = DocumentDownloadStage(
            downloader=self.downloader,
        )

        # Iterate stage
        iterate_stage = DocumentIterateStage(
            iterator=self.iterator,
            record_limit=self.record_limit,
            add_filename_column=self.add_filename_column,
        )

        # Extract stage (if extractor provided)
        stages = [url_stage, download_stage, iterate_stage]
        if self.extractor:
            extract_stage = DocumentExtractStage(
                extractor=self.extractor,
                add_filename_column=self.add_filename_column,
            )
            stages.append(extract_stage)

        self.stages = stages

    @property
    def name(self) -> str:
        """Return composite stage name."""
        generator_name = self.url_generator.__class__.__name__
        downloader_name = self.downloader.__class__.__name__
        return f"document_download_extract_{generator_name.lower()}_{downloader_name.lower()}_composite"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get description of this composite stage."""
        return f"URL-Download-Iterate-Extract pipeline using {self.url_generator.__class__.__name__} and {self.downloader.__class__.__name__}"
