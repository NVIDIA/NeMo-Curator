import json
import zlib
from datetime import date, datetime
from unittest.mock import Mock, patch

import pytest
import requests

from ray_curator.stages.download.text.common_crawl.url_generation import (
    MainCommonCrawlUrlStage,
    NewsCommonCrawlUrlStage,
)
from ray_curator.tasks import _EmptyTask


class TestMainCommonCrawlUrlStage:
    def test_post_init_sets_date_attributes(self):
        """Test that __post_init__ properly sets _start_date and _end_date attributes"""
        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        # After initialization, date attributes should be available
        assert hasattr(stage, "_start_date")
        assert hasattr(stage, "_end_date")
        assert stage._start_date == date(2021, 3, 8)  # 2021 week 10
        assert stage._end_date == date(2021, 3, 22)  # 2021 week 12

    def test_parse_datetime_from_snapshot_string_valid(self):
        """Test parsing valid snapshot strings"""
        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        # Test valid snapshot
        dt = stage._parse_datetime_from_snapshot_string("2021-10", for_start=True)
        assert dt.year == 2021
        assert dt.isocalendar()[1] == 10  # Week 10

    def test_parse_datetime_from_snapshot_string_invalid(self):
        """Test parsing invalid snapshot strings"""
        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        # Test invalid format
        with pytest.raises(ValueError, match="Invalid Main CC snapshot format"):
            stage._parse_datetime_from_snapshot_string("2021", for_start=True)

        # Test invalid week number
        with pytest.raises(ValueError, match="Week number must be between 1 and 53"):
            stage._parse_datetime_from_snapshot_string("2021-55", for_start=True)

    def test_start_end_dates_validation(self):
        """Test start/end date validation"""
        # Test start date after end date - validation happens during object creation
        with pytest.raises(ValueError, match="Start snapshot .* is after end snapshot"):
            MainCommonCrawlUrlStage(start_snapshot_str="2021-12", end_snapshot_str="2021-10")

    @patch("ray_curator.stages.download.text.common_crawl.url_generation.datetime")
    def test_future_end_date_adjustment(self, mock_datetime: Mock):
        """Test that future end dates are adjusted to today"""
        # Mock today's date to be 2021-06-15
        mock_datetime.now.return_value.date.return_value = date(2021, 6, 15)
        mock_datetime.fromisocalendar = datetime.fromisocalendar

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2022-10")

        # After __post_init__, the dates are available as attributes
        assert stage._end_date == date(2021, 6, 15)  # Should be adjusted to today
        assert stage._start_date == date(2021, 3, 8)  # 2021 week 10

    @patch("requests.get")
    def test_snapshot_index_success(self, mock_get: Mock):
        """Test successful fetching of snapshot index"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "CC-MAIN-2025-21",
                "name": "May 2025 Index",
                "timegate": "https://index.commoncrawl.org/CC-MAIN-2025-21/",
                "cdx-api": "https://index.commoncrawl.org/CC-MAIN-2025-21-index",
                "from": "2025-05-12T01:17:22",
                "to": "2025-05-25T06:44:06",
            },
            {
                "id": "CC-MAIN-2025-18",
                "name": "April 2025 Index",
                "timegate": "https://index.commoncrawl.org/CC-MAIN-2025-18/",
                "cdx-api": "https://index.commoncrawl.org/CC-MAIN-2025-18-index",
                "from": "2025-04-17T13:50:10",
                "to": "2025-05-01T01:05:29",
            },
        ]
        mock_get.return_value = mock_response

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")
        index = stage._snapshot_index

        assert len(index) == 2
        assert index[0]["id"] == "CC-MAIN-2025-21"
        assert index[1]["id"] == "CC-MAIN-2025-18"
        mock_get.assert_called_once_with("https://index.commoncrawl.org/collinfo.json", timeout=10)

    @patch("requests.get")
    def test_snapshot_index_request_failure(self, mock_get: Mock):
        """Test request failure when fetching snapshot index"""
        mock_get.side_effect = requests.RequestException("Network error")

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        with pytest.raises(RuntimeError, match="Failed to fetch Common Crawl index"):
            _ = stage._snapshot_index

    @patch("requests.get")
    def test_snapshot_index_json_decode_error(self, mock_get: Mock):
        """Test JSON decode error when fetching snapshot index"""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        with pytest.raises(RuntimeError, match="Failed to decode JSON"):
            _ = stage._snapshot_index

    @patch("requests.get")
    def test_generate_path_urls(self, mock_get: Mock):
        """Test generating path URLs from snapshot index"""
        # Mock the collinfo.json response
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "CC-MAIN-2021-21"},  # Should be excluded (after range)
            {"id": "CC-MAIN-2021-17"},
            {"id": "CC-MAIN-2021-10"},
            {"id": "CC-MAIN-2020-50"},  # Should be excluded (before range)
            {"id": "CC-MAIN-2008-2009"},  # Should be excluded (unsupported)
            {"id": "INVALID-FORMAT"},  # Should be excluded (invalid format)
        ]
        mock_get.return_value = mock_response

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-17")
        urls = stage.generate_path_urls()

        expected_urls = [
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-17/warc.paths.gz",
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/warc.paths.gz",
        ]
        assert urls == expected_urls

    @patch("requests.get")
    def test_generate_path_urls_old_snapshots(self, mock_get: Mock):
        """Test handling of old snapshots (before 2013)"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "CC-MAIN-2021-10"},
        ]
        mock_get.return_value = mock_response

        # Request data from 2010, should be adjusted to 2013
        stage = MainCommonCrawlUrlStage(start_snapshot_str="2010-01", end_snapshot_str="2021-10")

        with patch("ray_curator.stages.download.text.common_crawl.url_generation.logger") as mock_logger:
            _ = stage.generate_path_urls()
            mock_logger.warning.assert_called_once()
            assert "Adjusting start date to 2013-01-01" in str(mock_logger.warning.call_args)

    @patch("requests.get")
    def test_generate_data_urls_success(self, mock_get: Mock):
        """Test successful generation of data URLs"""
        # Mock the warc.paths.gz content
        warc_paths_content = (
            "crawl-data/CC-MAIN-2021-10/segments/file1.warc.gz\ncrawl-data/CC-MAIN-2021-10/segments/file2.warc.gz\n"
        )
        compressed_content = zlib.compress(warc_paths_content.encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed_content
        mock_get.return_value = mock_response

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        path_urls = ["https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/warc.paths.gz"]
        data_urls = stage.generate_data_urls(path_urls)

        expected_urls = [
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/segments/file1.warc.gz",
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/segments/file2.warc.gz",
        ]
        assert data_urls == expected_urls

    @patch("requests.get")
    def test_generate_data_urls_with_limit(self, mock_get: Mock):
        """Test URL generation with limit"""
        warc_paths_content = "file1.warc.gz\nfile2.warc.gz\nfile3.warc.gz\n"
        compressed_content = zlib.compress(warc_paths_content.encode("utf-8"))

        mock_response = Mock()
        mock_response.content = compressed_content
        mock_get.return_value = mock_response

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12", limit=2)

        path_urls = ["https://data.commoncrawl.org/test.warc.paths.gz"]
        data_urls = stage.generate_data_urls(path_urls)

        assert len(data_urls) == 2

    @patch("requests.get")
    def test_generate_data_urls_error_handling(self, mock_get: Mock):
        """Test handling of various errors in generate_data_urls"""
        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")
        path_urls = ["https://data.commoncrawl.org/test.warc.paths.gz"]

        with patch("ray_curator.stages.download.text.common_crawl.url_generation.logger") as mock_logger:
            # Test network error
            mock_get.side_effect = requests.RequestException("Network error")
            data_urls = stage.generate_data_urls(path_urls)
            assert data_urls == []
            assert mock_logger.error.call_count == 1

            # Test compression error
            mock_logger.reset_mock()
            mock_response = Mock()
            mock_response.content = b"invalid compressed data"
            mock_get.side_effect = None
            mock_get.return_value = mock_response

            data_urls = stage.generate_data_urls(path_urls)
            assert data_urls == []
            assert mock_logger.error.call_count == 1

    def test_generate_data_urls_empty_path_urls(self):
        """Test generate_data_urls with empty path_urls"""
        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")

        data_urls = stage.generate_data_urls([])
        assert data_urls == []

    @patch.object(MainCommonCrawlUrlStage, "generate_path_urls")
    @patch.object(MainCommonCrawlUrlStage, "generate_data_urls")
    def test_process(self, mock_generate_data_urls: Mock, mock_generate_path_urls: Mock):
        """Test the process method"""
        mock_generate_path_urls.return_value = ["path1.gz", "path2.gz"]
        mock_generate_data_urls.return_value = [
            "https://data.commoncrawl.org/file1.warc.gz",
            "https://data.commoncrawl.org/file2.warc.gz",
        ]

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")
        task = _EmptyTask(task_id="test_task", dataset_name="test_dataset", data=None)

        result = stage.process(task)

        assert len(result) == 2
        assert result[0].task_id == "test_task_0"
        assert result[0].dataset_name == "test_dataset"
        assert result[0].data == ["https://data.commoncrawl.org/file1.warc.gz"]
        assert result[1].task_id == "test_task_1"
        assert result[1].data == ["https://data.commoncrawl.org/file2.warc.gz"]


class TestNewsCommonCrawlUrlStage:
    def test_parse_datetime_from_snapshot_string_valid(self):
        """Test parsing valid snapshot strings for news"""
        stage = NewsCommonCrawlUrlStage(start_snapshot_str="2021-04", end_snapshot_str="2021-10")

        # Test start date
        dt = stage._parse_datetime_from_snapshot_string("2021-04", for_start=True)
        assert dt.year == 2021
        assert dt.month == 4
        assert dt.day == 1

        # Test end date (should be last day of month)
        dt = stage._parse_datetime_from_snapshot_string("2021-04", for_start=False)
        assert dt.year == 2021
        assert dt.month == 4
        assert dt.day == 30  # April has 30 days

    def test_parse_datetime_from_snapshot_string_invalid(self):
        """Test parsing invalid snapshot strings for news"""
        stage = NewsCommonCrawlUrlStage(start_snapshot_str="2021-04", end_snapshot_str="2021-10")

        # Test invalid format
        with pytest.raises(ValueError, match="Invalid News CC snapshot format"):
            stage._parse_datetime_from_snapshot_string("2021", for_start=True)

        # Test invalid month - the error message will be wrapped in the general format error
        with pytest.raises(ValueError, match="Invalid News CC snapshot format"):
            stage._parse_datetime_from_snapshot_string("2021-13", for_start=True)

    def test_generate_path_urls(self):
        """Test generating path URLs for news data"""
        stage = NewsCommonCrawlUrlStage(start_snapshot_str="2021-04", end_snapshot_str="2021-06")

        urls = stage.generate_path_urls()

        expected_urls = [
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/06/warc.paths.gz",
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/05/warc.paths.gz",
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/04/warc.paths.gz",
        ]
        assert urls == expected_urls

    def test_generate_path_urls_early_date(self):
        """Test handling of dates before news data availability"""
        stage = NewsCommonCrawlUrlStage(start_snapshot_str="2015-01", end_snapshot_str="2021-06")

        with patch("ray_curator.stages.download.text.common_crawl.url_generation.logger") as mock_logger:
            urls = stage.generate_path_urls()
            mock_logger.warning.assert_called_once()
            assert "2016" in str(mock_logger.warning.call_args)

        # Should start from 2016-08 (minimum news date)
        assert "2016/08" in urls[-1]

    def test_generate_path_urls_cross_year(self):
        """Test generating URLs across year boundary"""
        stage = NewsCommonCrawlUrlStage(start_snapshot_str="2020-11", end_snapshot_str="2021-02")
        urls = stage.generate_path_urls()

        expected_urls = [
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/02/warc.paths.gz",
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2021/01/warc.paths.gz",
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2020/12/warc.paths.gz",
            "https://data.commoncrawl.org/crawl-data/CC-NEWS/2020/11/warc.paths.gz",
        ]
        assert urls == expected_urls

    def test_stage_metadata(self):
        """Test stage name and input/output specifications"""
        main_stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")
        news_stage = NewsCommonCrawlUrlStage(start_snapshot_str="2021-04", end_snapshot_str="2021-10")

        # Both stages should have the same name and specifications
        for stage in [main_stage, news_stage]:
            assert stage.name == "common_crawl_url_generation"
            assert stage.inputs() == ([], [])
            assert stage.outputs() == (["data"], [])


# Integration-style tests for realistic scenarios
class TestIntegrationScenarios:
    @patch("requests.get")
    def test_main_crawl_realistic_scenario(self, mock_get: Mock):
        """Test a realistic main crawl scenario"""
        # Mock the warc.paths.gz responses
        warc_content1 = (
            "crawl-data/CC-MAIN-2021-10/segments/file1.warc.gz\ncrawl-data/CC-MAIN-2021-10/segments/file2.warc.gz"
        )
        warc_content2 = "crawl-data/CC-MAIN-2021-12/segments/file3.warc.gz"

        def mock_get_side_effect(url: str, **kwargs) -> Mock:  # noqa: ARG001
            mock_response = Mock()
            if "collinfo.json" in url:
                # Mock the snapshot index response
                mock_response.json.return_value = [
                    {"id": "CC-MAIN-2021-10"},  # Week 10 2021
                    {"id": "CC-MAIN-2021-12"},  # Week 12 2021
                ]
                return mock_response
            elif "2021-10" in url:
                mock_response.content = zlib.compress(warc_content1.encode("utf-8"))
                return mock_response
            else:
                mock_response.content = zlib.compress(warc_content2.encode("utf-8"))
                return mock_response

        mock_get.side_effect = mock_get_side_effect

        stage = MainCommonCrawlUrlStage(start_snapshot_str="2021-10", end_snapshot_str="2021-12")
        task = _EmptyTask(task_id="integration_test", dataset_name="cc_main", data=None)

        result = stage.process(task)

        # Should have files from both snapshots (2 + 1 = 3 total)
        assert len(result) == 3  # 3 WARC files total
        assert all(task.dataset_name == "cc_main" for task in result)
        assert all("file" in task.data[0] for task in result)

    @patch("requests.get")
    def test_news_crawl_realistic_scenario(self, mock_get: Mock):
        """Test a realistic news crawl scenario"""
        warc_content = "crawl-data/CC-NEWS/2021/04/CC-NEWS-20210401004522-01022.warc.gz\ncrawl-data/CC-NEWS/2021/04/CC-NEWS-20210401014522-01023.warc.gz"

        mock_response = Mock()
        mock_response.content = zlib.compress(warc_content.encode("utf-8"))
        mock_get.return_value = mock_response

        stage = NewsCommonCrawlUrlStage(start_snapshot_str="2021-04", end_snapshot_str="2021-04")
        task = _EmptyTask(task_id="news_test", dataset_name="cc_news", data=None)

        result = stage.process(task)

        assert len(result) == 2
        assert all(task.dataset_name == "cc_news" for task in result)
        assert all("CC-NEWS" in task.data[0] for task in result)
