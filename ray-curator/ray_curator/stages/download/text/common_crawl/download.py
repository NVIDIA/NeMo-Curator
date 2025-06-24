import os
import subprocess
from urllib.parse import urlparse

from loguru import logger

from ray_curator.stages.base import ProcessingStage
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
        urlpath = urlparse(url).path[1:]
        output_name = urlpath.replace("/", "-")
        output_file = os.path.join(self._download_dir, output_name)
        if os.path.exists(output_file):
            if self._verbose:
                logger.info(f"WARC file: {output_file} exists. Not downloading")
            return output_file

        url_to_download = os.path.join("s3://commoncrawl/", urlpath) if self.use_aws_to_download else url

        if self._verbose:
            logger.info(f"Downloading {url_to_download} and writing to {output_file}")
        # Download with either wget or s5cmd (aws)
        if self.use_aws_to_download:
            cmd = ["s5cmd", "cp", url_to_download, output_file]
        else:
            cmd = ["wget", url_to_download, "-O", output_file]

        # Always capture stderr so we can provide meaningful error messages
        if self._verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.PIPE
        p = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=stdout,
            stderr=stderr,
        )
        if p.returncode != 0:
            # Get the error message from stderr if available
            error_msg = p.stderr.decode("utf-8") if p.stderr else "Unknown error"
            logger.error(f"Failed to download {url_to_download} to {output_file} due to {error_msg}")
            return None

        return output_file

    @property
    def name(self) -> str:
        return "common_crawl_warc_downloader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
