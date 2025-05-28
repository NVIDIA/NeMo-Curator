"""Image-specific filter stages."""

import logging

from ray_curator.data import ImageBatch
from ray_curator.stages.base import ProcessingStage, StageType

logger = logging.getLogger(__name__)


class ImageSizeFilterStage(ProcessingStage[ImageBatch]):
    """Filter images based on size constraints."""

    def __init__(self, min_size: tuple[int, int] | None = None, max_size: tuple[int, int] | None = None):
        """Initialize the image size filter.

        Args:
            min_size: Minimum (width, height) in pixels
            max_size: Maximum (width, height) in pixels
        """
        self.min_size = min_size
        self.max_size = max_size

    @property
    def name(self) -> str:
        return "image_size_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    def process(self, batch: ImageBatch) -> ImageBatch | None:
        """Filter images by size."""
        # TODO: Implement image size filtering
        logger.info(f"Processing {len(batch.data)} images")
        return batch


class ImageFormatFilterStage(ProcessingStage[ImageBatch]):
    """Filter images based on format."""

    def __init__(self, allowed_formats: list[str]):
        """Initialize the image format filter.

        Args:
            allowed_formats: List of allowed formats (e.g., ["jpg", "png"])
        """
        self.allowed_formats = [fmt.lower() for fmt in allowed_formats]

    @property
    def name(self) -> str:
        return "image_format_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    def process(self, batch: ImageBatch) -> ImageBatch | None:
        """Filter images by format."""
        # TODO: Implement image format filtering
        logger.info(f"Processing {len(batch.data)} images")
        return batch
