from pathlib import Path
from typing import Literal

import pytest

from ray_curator.stages.download.text.common_crawl.download import CommonCrawlWARCDownloader
from ray_curator.stages.download.text.common_crawl.html_extractor import CommonCrawlHTMLExtractor
from ray_curator.stages.download.text.common_crawl.html_extractors import JusTextExtractor, ResiliparseExtractor
from ray_curator.stages.download.text.common_crawl.stage import CommonCrawl
from ray_curator.stages.download.text.common_crawl.url_generation import (
    MainCommonCrawlUrlStage,
    NewsCommonCrawlUrlStage,
)
from ray_curator.stages.download.text.common_crawl.warc_reader import WarcReader


class TestCommonCrawlStage:
    @pytest.mark.parametrize(
        ("crawl_type", "start_snapshot", "end_snapshot"),
        [
            ("main", "2021-23", "2021-26"),  # YYYY-WW format for main
            ("news", "2021-04", "2021-10"),  # YYYY-MM format for news
        ],
    )
    def test_common_crawl_stage_decomposition(
        self, tmp_path: Path, crawl_type: Literal["main", "news"], start_snapshot: str, end_snapshot: str
    ) -> None:
        """Test that CommonCrawl stage can be decomposed into constituent stages."""
        download_dir = str(tmp_path / "downloads")
        stage = CommonCrawl(
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            download_dir=download_dir,
            crawl_type=crawl_type,
            html_extraction="justext",
            limit=5,
        )

        # Decompose the stage
        stages = stage.decompose()

        # Should have 4 stages: URL generation, download, WARC reader, HTML extractor
        assert len(stages) == 4

        # Check stage names
        stage_names = [s.name for s in stages]
        expected_names = [
            "common_crawl_url_generation",  # URL generation
            "common_crawl_warc_downloader",  # Download
            "warc_processor",  # WARC reader
            "common_crawl_html_extractor",  # HTML extractor
        ]
        assert stage_names == expected_names

        # Verify the correct URL generation stage is used based on crawl_type
        if crawl_type == "main":
            assert isinstance(stages[0], MainCommonCrawlUrlStage)
        else:  # news
            assert isinstance(stages[0], NewsCommonCrawlUrlStage)

        # Verify other stages are correct types
        assert isinstance(stages[1], CommonCrawlWARCDownloader)
        assert isinstance(stages[2], WarcReader)
        assert isinstance(stages[3], CommonCrawlHTMLExtractor)

    def test_common_crawl_html_extraction_algorithms(self, tmp_path: Path) -> None:
        """Test different HTML extraction algorithms initialization."""
        download_dir = str(tmp_path / "downloads")

        # Test with string algorithm
        stage_justext = CommonCrawl(
            start_snapshot="2021-04", end_snapshot="2021-10", download_dir=download_dir, html_extraction="justext"
        )

        # Get the HTML extractor stage (4th stage)
        html_stage = stage_justext.decompose()[3]
        assert isinstance(html_stage, CommonCrawlHTMLExtractor)
        assert isinstance(html_stage.algorithm, JusTextExtractor)

        # Test with algorithm object and custom stop lists
        custom_stop_lists = {"ENGLISH": frozenset(["the", "and", "or"])}
        stage_resiliparse = CommonCrawl(
            start_snapshot="2021-04",
            end_snapshot="2021-10",
            download_dir=download_dir,
            html_extraction=ResiliparseExtractor(),
            stop_lists=custom_stop_lists,
        )

        html_stage = stage_resiliparse.decompose()[3]
        assert isinstance(html_stage, CommonCrawlHTMLExtractor)
        assert isinstance(html_stage.algorithm, ResiliparseExtractor)
        assert html_stage._stop_lists == custom_stop_lists
