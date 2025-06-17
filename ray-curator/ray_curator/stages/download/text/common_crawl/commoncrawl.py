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
from typing import Literal

from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    download_and_extract,
)
from nemo_curator.utils.download_utils import get_common_crawl_urls
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


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
