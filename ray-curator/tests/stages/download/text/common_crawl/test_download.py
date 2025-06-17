import os
import subprocess
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

from pytest import MonkeyPatch

from ray_curator.stages.download.text.common_crawl.download import CommonCrawlWARCDownloader
from ray_curator.tasks import FileGroupTask


class TestCommonCrawlWARCDownloader:
    """Test suite for CommonCrawlWARCDownloader."""

    def test_common_crawl_downloader_existing_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test that downloader skips downloading when file already exists."""
        # Create a temporary downloads directory and simulate an already-downloaded file.
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")  # "commoncrawl.warc"
        file_path = os.path.join(str(download_dir), output_name)
        # Write dummy content to simulate an existing download.
        with open(file_path, "w") as f:
            f.write("existing content")

        downloader = CommonCrawlWARCDownloader(str(download_dir), aws=False, verbose=False)

        # Monkey-patch subprocess.run to track if it gets called.
        called_run = False

        def fake_run(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
            nonlocal called_run
            called_run = True
            return mock.Mock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = downloader.download(url)
        assert result == file_path
        # Since the file already exists, no download should be attempted.
        assert not called_run

    def test_common_crawl_downloader_new_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test that downloader attempts to download when file doesn't exist."""
        # Create a temporary downloads directory; ensure the file does not exist.
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")  # "commoncrawl.warc"
        file_path = os.path.join(str(download_dir), output_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        downloader = CommonCrawlWARCDownloader(str(download_dir), aws=False, verbose=False)

        called_run = False

        def fake_run(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
            nonlocal called_run
            called_run = True
            return mock.Mock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = downloader.download(url)
        assert result == file_path
        # Since the file did not exist, a download call (and subprocess.run) should have been made.
        assert called_run

    def test_process_method(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test the process method with FileGroupTask."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        urls = ["http://dummy/file1.warc", "http://dummy/file2.warc"]

        task = FileGroupTask(task_id="test_task", dataset_name="test_dataset", data=urls, _stage_perf=[], _metadata={})

        downloader = CommonCrawlWARCDownloader(str(download_dir), aws=False, verbose=False)

        def fake_run(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
            return mock.Mock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = downloader.process(task)

        # Verify result structure
        assert isinstance(result, FileGroupTask)
        assert result.task_id == task.task_id
        assert result.dataset_name == task.dataset_name
        assert len(result.data) == 2
        assert "source_files" in result._metadata
        assert len(result._metadata["source_files"]) == 2

    def test_downloader_properties(self, tmp_path: Path) -> None:
        """Test downloader properties and methods."""
        download_dir = tmp_path / "downloads"
        downloader = CommonCrawlWARCDownloader(str(download_dir), aws=False, verbose=False)

        assert downloader.name == "common_crawl_warc_downloader"

        inputs = downloader.inputs()
        assert inputs == (["data"], [])

        outputs = downloader.outputs()
        assert outputs == (["data"], [])
