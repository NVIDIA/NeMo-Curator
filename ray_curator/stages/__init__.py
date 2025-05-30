"""Processing stages for the ray-curator pipeline."""

from .base import ProcessingStage, StageType

# Download stages
# from .download import CommonCrawlDownloadStage

# Image stages
# from .image import ImageFormatFilterStage, ImageSizeFilterStage

# Reader stages (internal use)
from .readers import JsonlProcessingStage, ParquetProcessingStage

# Text stages - commented out due to missing dependencies
# from .text import (
#     HtmlTextExtractorStage,
#     TextContentFilterStage,
#     TextLanguageFilterStage,
#     TextLengthFilterStage,
#     TextMetadataFilterStage,
# )

# Logical (composite) stages
# Optional logical (composite) stages are **not** imported here by default
# to avoid pulling in heavy optional dependencies like trafilatura or
# requests.  Users can import them explicitly, e.g.::
#
#     from ray_curator.stages.logical import Download

__all__ = [
    # Base classes
    "ProcessingStage",
    "StageType",
    # Reader stages (internal use)
    "JsonlProcessingStage",
    "ParquetProcessingStage",
    # Text filter stages
    # "TextLengthFilterStage",
    # "TextLanguageFilterStage",
    # "TextContentFilterStage",
    # "TextMetadataFilterStage",
    # Text extractor stages
    # "HtmlTextExtractorStage",
    # Image filter stages
    # "ImageSizeFilterStage",
    # "ImageFormatFilterStage",
    # Download stages
    # "CommonCrawlDownloadStage",
    # Logical (composite) stages
    # "Download",
]
