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
import warnings
from collections.abc import Iterator
from copy import deepcopy
from typing import Literal
from urllib.parse import urlparse

import justext
from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)
from nemo_curator.utils.download_utils import get_common_crawl_urls
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from resiliparse.extract.html2text import extract_plain_text
from trafilatura import extract as extract_with_trafilatura
from trafilatura.settings import DEFAULT_CONFIG as TRAFILATURA_DEFAULT_CONFIG
from warcio.archiveiterator import ArchiveIterator



class TrafilaturaExtractor(HTMLExtractorAlgorithm):
    def __init__(  # noqa: PLR0913
        self,
        required_stopword_density: float = 0.32,
        min_extracted_size: int = 250,
        min_extracted_comm_size: int = 1,
        min_output_size: int = 1,
        min_output_comm_size: int = 1,
        max_tree_size: int | None = None,
        min_duplcheck_size: int = 100,
        max_repetitions: int = 2,
        **extract_kwargs,
    ):
        """
        Initialize the Trafilatura text extraction algorithm with specified parameters.

        The Trafilatura extraction process combines readability-lxml and jusText as fallbacks to ensure robustness.
        Trafilatura's own algorithm follows a cascade of rule-based filters and content heuristics:
            • Content Delimitation: Uses XPath expressions to exclude unwanted HTML elements (e.g., navigation bars) and focus on relevant content (e.g., article body).
                Extracted HTML nodes are analyzed for relevance based on element type, text length, and link density.
            • Fallback Mechanism: If extraction seems faulty, alternative algorithms are run as backups.
                These use heuristics like line length, text-to-markup ratio, and HTML depth to improve extraction.
                Outputs are compared, prioritizing longer extractions with fewer impurities.
            • Baseline Extraction: If all else fails, it searches for text elements that might have been missed, discarding irrelevant content.

        The system balances precision and recall, extracting main text, comments, and metadata (title, site name, author, date, categories, tags).

        Please refer to the Trafilatura documentation for more details:
            https://trafilatura.readthedocs.io/en/latest/ and https://aclanthology.org/2021.acl-demo.15/

        NeMo Curator has added a stopword density filter to the Trafilatura extraction process, which requires that a paragraph contains a certain proportion of stopwords.

        Args:
            required_stopword_density: Proportion of stopwords required preserve an extracted paragraph.
                Studies on stopword lists and their distribution in various text corpora often
                suggest that around 30-40% of a typical English text consists of stopwords.
            min_extracted_size: Acceptable size in characters (used to trigger fallbacks).
                Defaults to 250. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_extracted_comm_size: Works the same as min_output_comm_size for comment extraction.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_output_size: Absolute acceptable minimum for main text output.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_output_comm_size: Works the same as min_output_comm_size for comment extraction.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            max_tree_size: Used to discard documents with too many elements. Defaults to None.
            min_duplcheck_size: Minimum size in characters to run deduplication on.
                Defaults to 100. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            max_repetitions: Maximum number of duplicates allowed.
                Defaults to 2. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            extract_kwargs: Additional keyword arguments for the Trafilatura extract function.
                See API documentation https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract
                for list of possible parameters.
                All arguments are set to their default values, except for deduplicate (bool) which is set to True.

        """
        self.required_stopword_density = required_stopword_density
        self.min_extracted_size = min_extracted_size
        self.min_extracted_comm_size = min_extracted_comm_size
        self.min_output_size = min_output_size
        self.min_output_comm_size = min_output_comm_size
        self.max_tree_size = max_tree_size
        self.min_duplcheck_size = min_duplcheck_size
        self.max_repetitions = max_repetitions
        self.extract_kwargs = extract_kwargs

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        trafilatura_config = deepcopy(TRAFILATURA_DEFAULT_CONFIG)
        trafilatura_config["DEFAULT"]["MIN_EXTRACTED_SIZE"] = str(self.min_extracted_size)
        trafilatura_config["DEFAULT"]["MIN_EXTRACTED_COMM_SIZE"] = str(self.min_extracted_comm_size)
        trafilatura_config["DEFAULT"]["MIN_OUTPUT_SIZE"] = str(self.min_output_size)
        trafilatura_config["DEFAULT"]["MIN_OUTPUT_COMM_SIZE"] = str(self.min_output_comm_size)
        if self.max_tree_size:
            trafilatura_config["DEFAULT"]["MAX_TREE_SIZE"] = str(self.max_tree_size)
        trafilatura_config["DEFAULT"]["MIN_DUPLCHECK_SIZE"] = str(self.min_duplcheck_size)
        trafilatura_config["DEFAULT"]["MAX_REPETITIONS"] = str(self.max_repetitions)

        # Recommended to set deduplicate=True
        self.extract_kwargs.setdefault("deduplicate", True)

        text = extract_with_trafilatura(html, config=trafilatura_config, **self.extract_kwargs)

        if text is not None:
            paragraphs = list(filter(None, text.split("\n")))

            if language in NON_SPACED_LANGUAGES:
                warnings.warn("stopword_density is ignored for non-space-separated languages.", stacklevel=2)
                result = paragraphs

            else:
                result = []

                for paragraph in paragraphs:
                    words = paragraph.split()
                    length = len(words)

                    if length == 0:
                        continue

                    stopwords = [word for word in words if word in stop_words]
                    stopword_density = len(stopwords) / length

                    if stopword_density >= self.required_stopword_density:
                        result.append(paragraph)

        else:
            return None

        if len(result) == 0:
            return None
        return result


def get_stop_list_dict(languages: list[str] | None = None) -> dict[str, frozenset[str]]:
    if languages is None:
        languages = []

    # Name mapping for language names from CLD2 (values)
    # and jusText (keys)
    lang_map = {
        "Haitian": "HAITIAN_CREOLE",
        "Norwegian_Bokmal": "NORWEGIAN",
        "Norwegian_Nynorsk": "NORWEGIAN_N",
        "Waray_Waray": "WARAY_PHILIPPINES",
    }

    # List obtained from https://github.com/stopwords-iso/stopwords-ja
    from .ja_stopwords import ja_stopwords

    # List obtained from https://github.com/stopwords-iso/stopwords-th
    from .th_stopwords import th_stopwords

    # List obtained from https://github.com/stopwords-iso/stopwords-zh
    from .zh_stopwords import zh_stopwords

    custom_stopwords = {
        "THAI": th_stopwords,
        "CHINESE": zh_stopwords,
        "JAPANESE": ja_stopwords,
    }

    if len(languages) == 0:
        languages = justext.get_stoplists()

        # Remove Latin as it yields a lot of low quality documents
        languages = list(languages)
        languages.remove("Latin")

        # Manually add Thai, Chinese, and Japanese
        languages.append("THAI")
        languages.append("CHINESE")
        languages.append("JAPANESE")

        languages = frozenset(languages)

    stop_list_dict = {}
    for language in languages:
        lang_key = lang_map[language] if language in lang_map else language.upper()

        if lang_key in custom_stopwords:
            stop_list_dict[lang_key] = custom_stopwords[lang_key]
        else:
            stop_list_dict[lang_key] = justext.get_stoplist(language)

    return stop_list_dict


def get_all_stop_words() -> frozenset[str]:
    stop_words = set()
    for language in justext.get_stoplists():
        stop_words.update(justext.get_stoplist(language))

    return frozenset(stop_words)


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
