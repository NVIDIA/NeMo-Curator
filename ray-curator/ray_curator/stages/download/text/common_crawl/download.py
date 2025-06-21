import os
import subprocess
from urllib.parse import urlparse

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
)


class CommonCrawlWARCDownloader(DocumentDownloader):
    """
    Downloads WARC files from the Common Crawl
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

    def download(self, url: str) -> str:
        # Download each URL to the directory
        urlpath = urlparse(url).path[1:]
        output_name = urlpath.replace("/", "-")
        output_file = os.path.join(self._download_dir, output_name)
        if os.path.exists(output_file):
            print(f"WARC file: {output_file} exists. Not downloading")
        else:
            print(f"Downloading {url} and writing to {output_file}")
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
                print(f"Failed to download {url} to {output_file}")

        return output_file


class CommonCrawlWARCDownloaderExtractOnly(DocumentDownloader):
    """
    A 'dummy' downloader that simply puts pre-downloaded
    files on the queue
    """

    def __init__(self, aws: bool = False, verbose: bool = False):  # noqa: ARG002
        super().__init__()

    def download(self, url: str) -> str:
        print(f"Putting WARC file {url} on the queue for extraction")
        return url
