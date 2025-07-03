# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import gzip
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from typing import Literal

from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)
from nemo_curator.utils.download_utils import get_arxiv_urls
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
)

# The iterator and extractor code are in large part taken
# from the Red-Pajama repo
# https://github.com/togethercomputer/RedPajama-Data/tree/main/data_prep/arxiv


def _is_safe_path(path: str, base_path: str) -> bool:
    """
    Check if a path is safe for extraction (no path traversal).

    Args:
        path: The path to check
        base_path: The base directory for extraction

    Returns:
        True if the path is safe, False otherwise
    """
    # Normalize paths to handle different path separators and resolve '..' components
    full_path = os.path.normpath(os.path.join(base_path, path))
    base_path = os.path.normpath(base_path)

    # Check if the resolved path is within the base directory
    return os.path.commonpath([full_path, base_path]) == base_path


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    """
    Safely extract a tar file, preventing path traversal attacks.

    Args:
        tar: The TarFile object to extract
        path: The destination path for extraction

    Raises:
        ValueError: If any member has an unsafe path
    """
    for member in tar.getmembers():
        # Check for absolute paths
        if os.path.isabs(member.name):
            msg = f"Absolute path not allowed: {member.name}"
            raise ValueError(msg)

        # Check for path traversal attempts
        if not _is_safe_path(member.name, path):
            msg = f"Path traversal attempt detected: {member.name}"
            raise ValueError(msg)

        # Check for dangerous file types
        if member.isdev():
            msg = f"Device files not allowed: {member.name}"
            raise ValueError(msg)

        # For symlinks, check that the target is also safe
        if member.issym() or member.islnk():
            if os.path.isabs(member.linkname):
                msg = f"Absolute symlink target not allowed: {member.name} -> {member.linkname}"
                raise ValueError(msg)
            if not _is_safe_path(member.linkname, path):
                msg = f"Symlink target outside extraction directory: {member.name} -> {member.linkname}"
                raise ValueError(msg)

        # Extract the member
        tar.extract(member, path)


class ArxivDownloader(DocumentDownloader):
    def __init__(self, download_dir: str, verbose: bool = False):
        super().__init__()
        self._download_dir = download_dir
        self._verbose = verbose

    def download(self, tarfile: str) -> str:
        output_file = os.path.join(self._download_dir, tarfile)
        s3path = os.path.join("s3://arxiv/src", tarfile)
        if os.path.exists(output_file):
            print(f"tar file: {output_file} exists. Not downloading")
        else:
            print(f"Downloading {s3path} and writing to {output_file}")
            cmd = ["s5cmd", "--request-payer=requester", "cp", s3path, output_file]
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
                print(f"Failed to download {s3path} to {output_file}")

        return output_file


class ArxivIterator(DocumentIterator):
    def __init__(self, log_frequency: int = 1000):
        super().__init__()
        self._log_frequency = log_frequency
        self._counter = 0

    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], list[str]]]:
        self._counter = 0
        download_dir = os.path.split(file_path)[0]
        bname = os.path.split(file_path)[-1]
        with tempfile.TemporaryDirectory(dir=download_dir) as tmpdir, tarfile.open(file_path) as tf:
            # Use safe extraction instead of extractall to prevent path traversal attacks
            _safe_extract(tf, tmpdir)
            for _i, item in enumerate(get_all_files_paths_under(tmpdir)):
                if self._counter > 0 and self._counter % self._log_frequency == 0:
                    print(f"Extracted {self._counter} papers from {file_path}")
                self._counter += 1

                tex_files = self._tex_proj_loader(item)
                arxiv_id = os.path.splitext(os.path.split(item)[-1])[0]

                # get the arxiv id in the correct format
                try:
                    clean_arxiv_id = self._format_arxiv_id(arxiv_id)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARNING] failed to format arxiv id {arxiv_id}; exception={e}")
                    clean_arxiv_id = arxiv_id

                if tex_files is None:
                    continue

                yield {"id": clean_arxiv_id, "source_id": f"{bname}"}, tex_files

    def _tex_proj_loader(self, file_or_dir_path: str) -> list[str] | None:
        r"""function to load the tex files from a tar file or a gzip file. The
        function will return a tuple containing a list of tex files and the
        timestamp of the project.

        @param file_or_dir_path: path to the tar file or the gzip file

        @return: tuple containing a list of tex files and the timestamp of the
            project
        """
        files_and_content = []

        try:
            # if it is a directory, open it as a tarfile
            with tarfile.open(file_or_dir_path) as sub_tf:
                for member in sub_tf.getmembers():
                    if member.name.endswith(".tex"):
                        file_content = sub_tf.extractfile(member).read()

                        try:
                            file_content = file_content.decode("utf-8")
                        except UnicodeDecodeError:
                            self._logger.info(f"UnicodeDecodeError: {file_or_dir_path}")
                            return None

                        files_and_content.append(file_content)

        except tarfile.ReadError:
            # otherwise we try opening it as a gzip file
            try:
                with gzip.open(file_or_dir_path, "rb") as gz:
                    file_content = gz.read()
            except Exception as e:  # noqa: BLE001
                # all fails, we skip this file
                self._logger.info(f"[ERROR] {e}: {file_or_dir_path}")
                return None

            try:
                file_content = file_content.decode("utf-8")
            except UnicodeDecodeError:
                self._logger.info(f"UnicodeDecodeError: {file_or_dir_path}")
                return None

            files_and_content.append(file_content)

        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {e}: {file_or_dir_path}")
            return None

        return files_and_content

    def _format_arxiv_id(self, arxiv_id: str) -> str:
        r"""this function brings the raw arxiv-id into a format compliant with the
        specification from arxiv. This is used to create the url to the arxiv
        abstract page.

        - Format prior to March 2007:
            <archive>/YYMMNNN where N is a 3-digit number
        - Format after March 2007: <archive>/YYMM.NNNNN where N is a
          5 (or 6)-digit number

        References: https://info.arxiv.org/help/arxiv_identifier.html

        @param arxiv_id: raw arxiv id which can be in one of the
                         following formats:
                         - <archive><YY><MM><NNN>
                         - <YY><MM><NNNNN|NNNNNN>

        @return: formatted arxiv id
        """
        match = re.search(r"^([a-zA-Z-]*)([\d\.]+)$", arxiv_id)

        if match is None:
            msg = f"Invalid arxiv id: {arxiv_id}"
            raise ValueError(msg)

        if match.group(1) == "":
            return match.group(2)

        return f"{match.group(1)}/{match.group(2)}"


class ArxivExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, content: list[str]) -> dict[str, str] | None:
        if len(content) == 0:
            return None

        # build dictionaries that contain the definitions of all macros in all tex
        # files. This is later used to expand all macros used in the text with
        # their definitions, so that consistency among different authors is
        # ensured.

        non_arg_macros = {}
        for file_content in content:
            non_arg_macros.update(self._build_non_arg_macros_dict(file_content))

        # TODO: macros that take arguments are not supported yet
        arg_macros = {}

        # join multiple latex files with a newline character
        try:
            cleaned_latex_file_str = "\n".join(
                self._clean_tex_file(
                    file_content=file_content,
                    arg_macros=arg_macros,
                    non_arg_macros=non_arg_macros,
                )
                for file_content in content
            )
        except Exception:  # noqa: BLE001
            return None

        # Don't return meta
        if (cleaned_latex_file_str is not None) and (len(cleaned_latex_file_str) > 0):
            return {"text": cleaned_latex_file_str}

        return None

    def _clean_tex_file(self, file_content: str, arg_macros: dict[str, str], non_arg_macros: dict[str, str]) -> str:
        r"""function takes a tex file as input and returns a cleaned version. The
         cleaned version is a concatenation of the tex files with the
        following modifications:

        - remove all comments (i.e. all lines starting with %)
        - remove everything before the first section-like header
        - remove everything after the first occurrence of either \appendix or
            \bibliography
        - inline-expand definitions and macros

        @param file_content: the content of the tex file as a string.

        @return: cleaned tex file as a string
        """
        # find the first occurence of a \section-like header and replace everything
        # before it with an empty string. This matches the following pattern:
        #   \<section-type>[optional-args]{name}
        pattern = r"^(.*?)("
        pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r")"

        # if no section like header is found, then we return an empty string
        if not re.search(pattern, file_content, flags=re.DOTALL):
            return ""

        # replace everything with the second group of the match (i.e. everything
        # after and including the section header)
        file_content = re.sub(
            pattern=pattern,
            repl=r"\2",
            string=file_content,
            flags=re.DOTALL,  # make sure that the dot matches also newlines
        )

        # remove all line comments
        file_content = re.sub(
            pattern=r"(?m)^%.*\n?",
            repl=r"",
            string=file_content,
            flags=re.MULTILINE,
        )

        # remove all in comments within a line
        file_content = re.sub(
            # pattern matches a "%" that is not preceded by a backslash (=comment)
            pattern=r"[^\\]%.+$",
            repl=r"",
            string=file_content,
            flags=re.MULTILINE,
        )

        # find the first occurence of either \appendix or \bibliography and
        # replace everything after it with an empty string
        pattern = r"("
        pattern += r"\\appendix|"
        pattern += r"\\begin\{references\}|"
        pattern += r"\\begin\{REFERENCES\}|"
        pattern += r"\\begin\{thebibliography\}|"
        pattern += r"\\bibliography\{.*\}"
        pattern += r").*$"

        file_content = re.sub(
            pattern=pattern,
            repl=r"",
            string=file_content,
            flags=re.DOTALL,  # make sure that the dot matches also newlines
        )

        # inline-expand all non-arg macros
        for macro_name, macro_value in non_arg_macros.items():
            file_content = re.sub(
                # make pattern grouped to make sure that the macro is not part
                # of a longer alphanumeric word
                pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
                # replace the macro with its value and add back the character that
                # was matched after the macro
                repl=macro_value + r"\2",
                string=file_content,
            )

        # inline-expand all macros that use args
        # TODO: inline-expand macros with args
        for _macro_name, _macro_value in arg_macros.items():
            pass

        return file_content

    def _build_non_arg_macros_dict(self, file_content: str) -> dict[str, str]:
        r"""function takes the content of a tex file and returns a dictionary
        that contains the definitions of all macros that do not use arguments.
        The dictionary is of the form {macro_name: macro_value}.

        @param file_content: the content of the tex file as a string.

        @return: dict
        """
        # regex for extracting \newcommand macros without arguments
        non_arg_nc_reg = re.compile(
            # this regex matches the following:
            # \newcommand{\macro_name}{macro_value}
            # \newcommand*{\macro_name}{macro_value}
            # where macro_name is only allowed to contain letters and numbers;
            # macro_value can contain any character.
            pattern=r"\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$",
            flags=re.MULTILINE,
        )

        # regex for extracting \def macros without arguments
        non_arg_def_reg = re.compile(
            # this regex matches the following:
            # \def\macro_name{macro_value}
            # where macro_name is only allowed to contain letters and numbers;
            # macro_value can contain any character.
            pattern=r"\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$",
            flags=re.MULTILINE,
        )

        # Extract all user-defined LaTeX macros from the preamble
        macros = {}
        for reg in [non_arg_nc_reg, non_arg_def_reg]:
            for match in reg.finditer(file_content):
                # convert the macro name and value to a raw string that can be
                # used in re.sub
                macro_name = match.group(1).encode("unicode-escape").decode("utf-8")
                macro_val = match.group(2).encode("unicode-escape").decode("utf-8")

                macros[macro_name] = macro_val

        return macros


def download_arxiv(  # noqa: PLR0913
    output_path: str,
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    raw_download_dir: str | None = None,
    keep_raw_download: bool = False,
    force_download: bool = False,
    url_limit: int | None = None,
    record_limit: int | None = None,
) -> DocumentDataset:
    """
    Download Arxiv tar files and extract the contained LaTeX projects.

    This function obtains a list of Arxiv tar file URLs (via get_arxiv_urls), downloads the tar files,
    and then extracts the contained LaTeX source files. The resulting documents (after extraction) are
    assembled into a DocumentDataset.

    Args:
        output_path (str):
            The root directory where both the final extracted files and the raw download subdirectory will be stored.
            The extracted files (in the format specified by output_type) are eventually saved in this directory.
        output_type (Literal["jsonl", "parquet"], optional):
            The file format/extension used for saving the extracted documents (e.g., "jsonl" or "parquet").
            Default is "jsonl". This is not used for the output file, but is used to check if an extracted output already exists and read it if so.
        raw_download_dir (Optional[str], optional):
            The directory where the raw downloaded tar files will be kept. If None, a folder named "downloads"
            under output_path is used.
        keep_raw_download (bool, optional):
            If True, the raw tar files (before extraction) are not removed after processing. Default is False.
        force_download (bool, optional):
            If False, then if an output file already exists for a given URL, re-downloading and re-extraction will be skipped.
            Default is False.
        url_limit (Optional[int], optional):
            Limits the maximum number of Arxiv tar file URLs to download and process.
            If None, all available URLs (from get_arxiv_urls) are processed.
        record_limit (Optional[int], optional):
            Limits the maximum number of records to extract from each tar file.
            If None, all available records are extracted.
    Returns:
        DocumentDataset:
            A dataset object containing the extracted documents.
    """
    arxiv_urls = get_arxiv_urls()
    if url_limit:
        arxiv_urls = arxiv_urls[:url_limit]
    output_paths = [os.path.join(output_path, f"{url}.{output_type}") for url in arxiv_urls]

    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)
    downloader = ArxivDownloader(raw_download_dir)
    iterator = ArxivIterator()
    extractor = ArxivExtractor()

    output_format = {
        "text": str,
        "id": str,
        "source_id": str,
        "file_name": str,
    }

    return download_and_extract(
        arxiv_urls,
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
