# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
from collections.abc import Iterator
from typing import Literal
from urllib.parse import urlparse

from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)
from nemo_curator.utils.download_utils import get_common_crawl_urls
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from warcio.archiveiterator import ArchiveIterator


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


class CommonCrawlWARCIterator(DocumentIterator):
    def __init__(self, log_frequency: int = 1000):
        super().__init__()
        self._counter = 0
        self._log_frequency = log_frequency

    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], str]]:
        # Loop over all records in the current WARC
        self._counter = 0
        bname = os.path.split(file_path)[-1]
        with open(file_path, "rb") as file_pointer:
            ai = ArchiveIterator(file_pointer, arc2warc=True)
            for _k, rec in enumerate(ai):
                # Get the response from the crawl
                if rec.rec_type == "response":
                    if self._counter > 0 and self._counter % self._log_frequency == 0:
                        print(f"Extracted {self._counter} records in WARC")
                    self._counter += 1
                    content = rec.content_stream().read()
                    warc_id = rec.rec_headers.get_header("WARC-Record-ID")[10:-1]
                    url = rec.rec_headers.get_header("WARC-Target-URI")
                    meta = {
                        "url": url,
                        "warc_id": warc_id,
                        "source_id": f"{bname}",
                    }
                    yield meta, content


class CommonCrawlWARCExtractor(DocumentExtractor):
    def __init__(
        self,
        algorithm: HTMLExtractorAlgorithm | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
    ):
        if algorithm is None:
            algorithm = JusTextExtractor()

        if stop_lists is not None:
            self._stop_lists = stop_lists
        else:
            self._stop_lists = get_stop_list_dict()

        self.algorithm = algorithm
        super().__init__()

    def extract(self, content: str) -> dict[str, str] | None:
        html = decode_html(content)
        if html is not None:
            # Language detection and HTML extraction
            lang = lang_detect(html)
            text = None
            if lang in self._stop_lists:
                text = self.algorithm.extract_text(html, self._stop_lists[lang], lang)
            if text is not None:
                if len(text) > 0:
                    text = "\n\n".join(text)
                    return {"language": lang, "text": text}
                else:
                    return None
        return None


def download_common_crawl(  # noqa: PLR0913
    output_path: str,
    start_snapshot: str,
    end_snapshot: str,
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    algorithm: HTMLExtractorAlgorithm | None = None,
    stop_lists: dict[str, frozenset[str]] | None = None,
    news: bool = False,
    aws: bool = False,
    raw_download_dir: str | None = None,
    keep_raw_download: bool = False,
    force_download: bool = False,
    url_limit: int | None = None,
    record_limit: int | None = None,
) -> DocumentDataset:
    """
    Downloads Common Crawl WARC snapshots and extracts text content using a specified extraction algorithm.

    Args:
      output_path (str): The root directory used for managing download and extraction.
          • Raw WARC files are stored in a "downloads" subdirectory under this path.
          • This path is also checked for existing extraction results; if found, extraction can be skipped.
          • Note: This function returns a DocumentDataset, and writing the extracted data to disk is the caller's responsibility.
      start_snapshot (str): Identifier for the earliest snapshot to process.
          • For CC-MAIN datasets, use the format 'YYYY-WeekNumber' (e.g., '2020-50' or '2021-04').
          • For CC-NEWS datasets (when news=True), use the 'YYYY-MM' (Year-Month) format.
      end_snapshot (str): Identifier for the latest snapshot to process, which must be chronologically after start_snapshot.
      output_type (Literal["jsonl", "parquet"]): The file format for the extracted output. Must be either "jsonl" or "parquet".
          • This is not used for the output file, but is used to check if an extracted output already exists.
      algorithm: The text extraction algorithm instance to use for HTML processing.
          • This can be a JusTextExtractor (default), ResiliparseExtractor, or TrafilaturaExtractor object.
      stop_lists: A dictionary stop lists, where the keys are languages (e.g., "ENGLISH")
          and the values are Python frozensets denoting the list of stop words for that language.
          If None, it defaults to jusText's stop lists: https://github.com/miso-belica/jusText/tree/main/justext/stoplists,
          with added Thai, Chinese, and Japanese support.
      news (bool): When True, indicates that URLs should be retrieved from the CC-NEWS dataset.
          • This also means snapshot identifiers should follow the 'YYYY-MM' format.
      aws (bool): If True, downloads are sourced from Common Crawl's S3 bucket using s5cmd;
          • If False, wget is used to fetch the files via HTTPS.
      raw_download_dir: Optional; the directory to temporarily store raw WARC files.
          • If not provided, defaults to a "downloads" folder within output_path.
      keep_raw_download (bool): If True, retains the downloaded raw WARC files after extraction.
          • If False, these raw files may be removed following extraction.
      force_download (bool): If False, skips re-downloading or re-extracting snapshots if outputs already exist in output_path.
      url_limit: Optional; the maximum number of WARC files to download from the snapshot range.
          • If None, all available files within the specified snapshots are downloaded.
      record_limit: Optional; the maximum number of records to extract from each WARC file.
          • If None, all available records are extracted.
    """
    if algorithm is None:
        algorithm = JusTextExtractor()

    common_crawl_urls = get_common_crawl_urls(
        starting_snapshot=start_snapshot, ending_snapshot=end_snapshot, news=news
    )

    if len(common_crawl_urls) == 0:
        msg = (
            f"No Common Crawl download urls found between {start_snapshot} and {end_snapshot}. "
            "Ensure that a valid Common Crawl snapshot (https://data.commoncrawl.org/) is "
            "within the range provided."
        )
        raise ValueError(msg)

    if url_limit:
        common_crawl_urls = common_crawl_urls[:url_limit]
    output_paths = [os.path.join(output_path, url.split("/")[-1] + f".{output_type}") for url in common_crawl_urls]

    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)
    downloader = CommonCrawlWARCDownloader(raw_download_dir, aws=aws)
    iterator = CommonCrawlWARCIterator()
    extractor = CommonCrawlWARCExtractor(algorithm=algorithm, stop_lists=stop_lists)

    output_format = {
        "text": str,
        "language": str,
        "url": str,
        "warc_id": str,
        "source_id": str,
        "file_name": str,
    }

    return download_and_extract(
        common_crawl_urls,
        output_paths,
        downloader,
        iterator,
        extractor,
        output_format,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        filename_col="file_name",
        record_limit=record_limit,
    )
