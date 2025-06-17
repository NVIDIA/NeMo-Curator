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

    def __init__(self, download_dir: str, aws: bool = False, verbose: bool = False):
        """
        Creates a downloader

        Args:
          download_dir: Path to store raw compressed WARC files
          aws: If True, uses the s5cmd command to download from the Common Crawl's S3 bucket.
            If False, uses wget.
          verbose: If True, logs stdout and stderr of the download command (s5cmd/wget)
        """
        super().__init__()
        self._download_dir = download_dir
        self._aws = aws
        self._verbose = verbose
        os.makedirs(download_dir, exist_ok=True)  # TOOD: Should this be possible on Remote?

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

        if self._verbose:
            logger.info(f"Downloading {url} and writing to {output_file}")
        # Download with either wget or s5cmd (aws)
        if self._aws:
            s3path = os.path.join("s3://commoncrawl/", urlpath)
            cmd = ["s5cmd", "cp", s3path, output_file]
        else:
            cmd = ["wget", url, "-O", output_file]
        if self._verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
        p = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdout=stdout,
            stderr=stderr,
        )
        if p.returncode != 0:
            logger.error(f"Failed to download {url} to {output_file} due to {p.stderr}")
            return None

        return output_file

    @property
    def name(self) -> str:
        return "common_crawl_warc_downloader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []
