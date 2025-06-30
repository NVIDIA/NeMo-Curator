import subprocess
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import pytest

from ray_curator.stages.download.text.common_crawl.download import CommonCrawlWARCDownloader
from ray_curator.tasks import FileGroupTask


class TestCommonCrawlWARCDownloader:
    """Test suite for CommonCrawlWARCDownloader."""

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_wget(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with wget (use_aws_to_download=False)."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        result = downloader._download_to_path(url, temp_path)

        assert result.returncode == 0
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch.object(CommonCrawlWARCDownloader, "_check_s5cmd_installed", return_value=True)
    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_s3(self, mock_run: mock.Mock, mock_s5cmd_check: mock.Mock, tmp_path: Path) -> None:  # noqa: ARG002
        """Test _download_to_path with s5cmd (use_aws_to_download=True)."""

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=True, verbose=False)

        url = "http://dummy/crawl-data/CC-MAIN-2024-10/segments/1234567890/warc/CC-MAIN-123.warc.gz"
        temp_path = str(tmp_path / "temp_file.tmp")

        result = downloader._download_to_path(url, temp_path)

        assert result.returncode == 0
        expected_s3_url = "s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/1234567890/warc/CC-MAIN-123.warc.gz"
        mock_run.assert_called_once_with(
            ["s5cmd", "cp", expected_s3_url, temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_verbose_logging(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with verbose logging enabled."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=True)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        result = downloader._download_to_path(url, temp_path)

        assert result.returncode == 0
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=None,
            stderr=None,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_quiet(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with quiet mode (verbose=False)."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        result = downloader._download_to_path(url, temp_path)

        assert result.returncode == 0
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch.object(CommonCrawlWARCDownloader, "_download_to_path")
    def test_download_existing_file(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test that download skips downloading when file already exists and is non-empty."""

        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")
        file_path = tmp_path / output_name

        # Create existing file with content
        file_path.write_text("existing content")

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        result = downloader.download(url)

        assert result == str(file_path)
        mock_download.assert_not_called()  # Should not attempt download for existing file

    @mock.patch.object(CommonCrawlWARCDownloader, "_download_to_path")
    def test_download_empty_existing_file(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test that download attempts to download when existing file is empty."""
        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")
        file_path = tmp_path / output_name

        # Create empty existing file
        file_path.touch()
        assert file_path.exists()
        assert file_path.stat().st_size == 0

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        def create_temp_file(url: str, path: str) -> mock.Mock:  # noqa: ARG001
            Path(path).write_text("downloaded content")
            return mock.Mock(returncode=0)

        mock_download.side_effect = create_temp_file

        result = downloader.download(url)

        assert result == str(file_path)
        assert file_path.read_text() == "downloaded content"
        mock_download.assert_called_once()

    @mock.patch.object(CommonCrawlWARCDownloader, "_download_to_path")
    def test_download_new_file_success(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test successful download of a new file."""
        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")
        file_path = tmp_path / output_name

        # Ensure file doesn't exist
        if file_path.exists():
            file_path.unlink()

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        def create_temp_file(url: str, path: str) -> mock.Mock:  # noqa: ARG001
            Path(path).write_text("downloaded content")
            return mock.Mock(returncode=0)

        mock_download.side_effect = create_temp_file

        result = downloader.download(url)

        assert result == str(file_path)
        assert file_path.exists()
        assert file_path.read_text() == "downloaded content"
        mock_download.assert_called_once()

    @mock.patch.object(
        CommonCrawlWARCDownloader, "_download_to_path", return_value=mock.Mock(returncode=1, stderr=b"Download failed")
    )
    def test_download_failure(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test download failure handling."""

        url = "http://dummy/commoncrawl.warc"

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        result = downloader.download(url)

        assert result is None
        mock_download.assert_called_once()

    @mock.patch.object(CommonCrawlWARCDownloader, "_download_to_path")
    def test_download_cleans_up_existing_temp_file(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test that download cleans up existing temporary files before attempting download."""

        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")
        temp_file_path = tmp_path / (output_name + ".tmp")

        # Create existing temp file
        temp_file_path.write_text("old temp content")
        assert temp_file_path.exists()

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        def create_new_temp_file(url: str, path: str) -> mock.Mock:  # noqa: ARG001
            # Temp file should have been cleaned up before this call
            assert not Path(path).exists() or Path(path).read_text() != "old temp content"
            Path(path).write_text("new downloaded content")
            return mock.Mock(returncode=0)

        mock_download.side_effect = create_new_temp_file

        result = downloader.download(url)

        file_path = tmp_path / output_name
        assert result == str(file_path)
        assert file_path.read_text() == "new downloaded content"
        assert not temp_file_path.exists()  # Temp file should be cleaned up
        mock_download.assert_called_once()

    @mock.patch.object(CommonCrawlWARCDownloader, "_download_to_path")
    def test_process_method(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test the process method with FileGroupTask."""

        urls = ["http://dummy/file1.warc", "http://dummy/file2.warc"]
        task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=urls, _stage_perf=[], _metadata={})

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        def create_temp_file(url: str, path: str) -> mock.Mock:
            Path(path).write_text(f"content for {url}")
            return mock.Mock(returncode=0)

        mock_download.side_effect = create_temp_file

        result = downloader.process(task)

        # Verify result structure
        assert isinstance(result, FileGroupTask)
        assert result.task_id == task.task_id
        assert result.dataset_name == task.dataset_name
        assert len(result.data) == 2
        assert "source_files" in result._metadata
        assert len(result._metadata["source_files"]) == 2

        # Verify files were created
        for file_path in result.data:
            assert Path(file_path).exists()

        # Verify download was called twice
        assert mock_download.call_count == 2

    @mock.patch.object(CommonCrawlWARCDownloader, "_download_to_path")
    def test_process_method_with_failures(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test the process method handling some failed downloads."""

        urls = ["http://dummy/file1.warc", "http://dummy/file2.warc", "http://dummy/file3.warc"]
        task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=urls, _stage_perf=[], _metadata={})

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        def mixed_download(url: str, path: str) -> mock.Mock:
            if "file2" in url:
                return mock.Mock(returncode=1, stderr=b"Download failed")
            else:
                Path(path).write_text(f"content for {url}")
                return mock.Mock(returncode=0)

        mock_download.side_effect = mixed_download

        result = downloader.process(task)

        # Should only have 2 successful downloads (file1 and file3)
        assert len(result.data) == 2
        assert len(result._metadata["source_files"]) == 2

        # Verify download was called thrice
        assert mock_download.call_count == 3

    def test_downloader_properties(self, tmp_path: Path) -> None:
        """Test downloader properties and methods."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        assert downloader.name == "common_crawl_warc_downloader"

        inputs = downloader.inputs()
        assert inputs == (["data"], [])

        outputs = downloader.outputs()
        assert outputs == (["data"], [])

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_check_s5cmd_installed_success(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _check_s5cmd_installed when s5cmd is available."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        assert downloader._check_s5cmd_installed() is True

        # Verify the correct command was called
        mock_run.assert_called_once_with(
            ["s5cmd", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )

    @mock.patch("subprocess.run", side_effect=FileNotFoundError("s5cmd not found"))
    def test_check_s5cmd_installed_failure(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _check_s5cmd_installed when s5cmd is not available."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        assert downloader._check_s5cmd_installed() is False
        mock_run.assert_called_once()

    @mock.patch.object(CommonCrawlWARCDownloader, "_check_s5cmd_installed", return_value=False)
    def test_s5cmd_check_during_init(self, mock_s5cmd_check: mock.Mock, tmp_path: Path) -> None:
        """Test that constructor checks for s5cmd when use_aws_to_download=True."""

        with pytest.raises(RuntimeError, match="s5cmd is not installed"):
            CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=True, verbose=False)

        mock_s5cmd_check.assert_called_once()

    def test_download_with_context_manager_example(self, tmp_path: Path) -> None:
        """Example of using context manager for very specific scoped patches."""

        url = "http://dummy/commoncrawl.warc"
        file_path = tmp_path / "commoncrawl.warc"
        file_path.write_text("existing content")  # File exists

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        # Use context manager when you need very specific control over patch scope
        with mock.patch.object(downloader, "_download_to_path") as mock_download:
            result = downloader.download(url)

            # Should return existing file without calling download
            assert result == str(file_path)
            mock_download.assert_not_called()

        # Outside context, the method is back to normal
        assert hasattr(downloader, "_download_to_path")  # Method exists normally
