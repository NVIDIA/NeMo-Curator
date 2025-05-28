"""Processing stages for the ray-curator pipeline."""

from .base import ProcessingStage, StageType

# Download stages
from .download import CommonCrawlDownloadStage

# Image stages
from .image import ImageFormatFilterStage, ImageSizeFilterStage

# Reader stages (internal use)
from .readers import JsonlProcessingStage, ParquetProcessingStage

# Text stages
from .text import (
    HtmlTextExtractorStage,
    TextContentFilterStage,
    TextLanguageFilterStage,
    TextLengthFilterStage,
    TextMetadataFilterStage,
)

__all__ = [
    # Base classes
    "ProcessingStage",
    "StageType",
    # Reader stages (internal use)
    "JsonlProcessingStage",
    "ParquetProcessingStage",
    # Text filter stages
    "TextLengthFilterStage",
    "TextLanguageFilterStage",
    "TextContentFilterStage",
    "TextMetadataFilterStage",
    # Text extractor stages
    "HtmlTextExtractorStage",
    # Image filter stages
    "ImageSizeFilterStage",
    "ImageFormatFilterStage",
    # Download stages
    "CommonCrawlDownloadStage",
]
