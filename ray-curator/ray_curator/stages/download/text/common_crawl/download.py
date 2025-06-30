import os
import subprocess
from urllib.parse import urlparse

from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask


class CommonCrawlWARCDownloader(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Downloads WARC files from the Common Crawl to a local directory
    """

    def __init__(self, download_dir: str, use_aws_to_download: bool = False, verbose: bool = False):
        """
        Creates a downloader

        Args:
          download_dir: Path to store raw compressed WARC files
          use_aws_to_download: If True, uses the s5cmd command to download from the Common Crawl's S3 bucket.
            If False, uses wget.
          verbose: If True, logs stdout and stderr of the download command (s5cmd/wget)
        """
        super().__init__()
        self._download_dir = download_dir
        self.use_aws_to_download = use_aws_to_download
        self._verbose = verbose
        os.makedirs(download_dir, exist_ok=True)  # TOOD: Should this be possible on Remote?
        if self.use_aws_to_download and not self._check_s5cmd_installed():
            msg = "s5cmd is not installed. Please install it from https://github.com/peak/s5cmd"
            raise RuntimeError(msg)

    def _check_s5cmd_installed(self) -> bool:
        """Check if s5cmd is installed."""
        try:
            subprocess.run(["s5cmd", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)  # noqa: S603, S607
        except FileNotFoundError:
            return False
        else:
            return True

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process a task containing WARC URLs and download them."""
        downloaded_files = []
        for url in task.data:
            output_file = self.download(url)
            if output_file:
                downloaded_files.append(output_file)

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=downloaded_files,
            _stage_perf=task._stage_perf,
            # this overrides the source_files metadata from the previous stage i.e BaseCommonCrawlUrlStage
            _metadata={"source_files": downloaded_files},
        )

    def download(self, url: str) -> str | None:
        # Download each URL to the directory
        output_name = urlparse(url).path[1:].replace("/", "-")
        output_file = os.path.join(self._download_dir, output_name)
        temp_file = output_file + ".tmp"

        # If final file exists and is non-empty, assume it's complete
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            if self._verbose:
                logger.info(f"WARC file: {output_file} exists. Not downloading")
            return output_file

        # Clean up any existing temporary file from previous failed attempts
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Download to temporary file
        result = self._download_to_path(url, temp_file)

        if result.returncode == 0:
            # Download successful, atomically move temp file to final location
            os.rename(temp_file, output_file)
            if self._verbose:
                file_size = os.path.getsize(output_file)
                logger.info(f"Successfully downloaded to {output_file} ({file_size} bytes)")
            return output_file
        else:
            # Download failed
            error_msg = result.stderr.decode("utf-8") if result.stderr else "Unknown error"
            logger.error(f"Failed to download to {output_file}: {error_msg}")
            return None

    def _download_to_path(self, url: str, path: str) -> subprocess.CompletedProcess:
        """Download a file to a temporary file."""
        urlpath = urlparse(url).path[1:]

        url_to_download = os.path.join("s3://commoncrawl/", urlpath) if self.use_aws_to_download else url

        if self._verbose:
            logger.info(f"Downloading {url_to_download} to {path}")

        # Download with either wget or s5cmd (aws) to temporary file
        if self.use_aws_to_download:
            cmd = ["s5cmd", "cp", url_to_download, path]
        else:
            cmd = ["wget", url_to_download, "-O", path]

        # Always capture stderr so we can provide meaningful error messages
        if self._verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.PIPE

        return subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=stdout,
            stderr=stderr,
        )

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        return Resources(cpus=0.5)

    @property
    def name(self) -> str:
        return "common_crawl_warc_downloader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
