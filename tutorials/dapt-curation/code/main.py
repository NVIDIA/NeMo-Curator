# +
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

import argparse
import os
import shutil
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
from downloaders import (
    download_github_sources,
    download_pdf_sources,
    download_wikipedia_sources,
)
from utils import (
    clean_and_unify,
    dedupe,
    filter_code,
    filter_code_dataset,
    filter_code_lines,
    filter_text,
    filter_text_lines,
    redact_code,
)

import nemo_curator as nc
from nemo_curator import ExactDuplicates, Modify, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import RepeatingTopNGramsFilter, WordCountFilter
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import (
    get_all_files_paths_under,
    separate_by_metadata,
)
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")


def download_sources(
    wikipedia_limit: Optional[int] = None,
    github_limit: Optional[int] = None,
    pdf_limit: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Downloads all the dataset sources and converts them to the JSONL format.

    Returns:
        tuple: the list of text files and the list of code files.
    """

    wikipedia_dir = download_wikipedia_sources(
        "sources/wikipedia_urls.jsonl", limit=wikipedia_limit
    )
    github_dir = download_github_sources(
        "sources/github_repos.jsonl", limit=github_limit
    )
    pdf_dir = download_pdf_sources("sources/arxiv_urls.jsonl", limit=pdf_limit)

    wiki_files = get_all_files_paths_under(wikipedia_dir)
    code_files = get_all_files_paths_under(github_dir)
    pdf_files = get_all_files_paths_under(pdf_dir)

    text_files = wiki_files + pdf_files

    return text_files, code_files


def plot_data(orig_dataset: DocumentDataset, filename: str):
    """
    Plot histogram of different file types and corresponding sizes

    Args:
        dataset (DocumentDataset): Dataset
        filename (str): Name of the plot to be saved ('sample.png')
    """
    # visualize file types and sizes
    orig_df = orig_dataset.df.compute()
    orig_df = orig_df.reset_index()

    # Create a histogram for different file types -text
    fig, ax = plt.subplots(figsize=(10, 6))
    orig_df.groupby("file_extension")["size_in_bytes"].sum().plot(kind="bar", ax=ax)
    ax.set_xlabel("file_extension")
    ax.set_ylabel("size_in_bytes")
    ax.set_title("File Size Histogram by File Extension")

    # Save the histogram to a file
    fig.savefig(filename, bbox_inches="tight")


def run_curation_pipeline(args: Any, text_files: str, code_files: str) -> None:
    """
    Run the curation pipeline on the TinyStories dataset.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    print("Running the curation pipeline...")
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))

    # Define data curation steps for text and pdf files
    curation_steps_text = Sequential(
        [
            dedupe,
            filter_text_lines,
            filter_text,
            clean_and_unify,
        ]
    )

    # Define data curation steps for code files
    curation_steps_code = Sequential(
        [
            dedupe,
            filter_code_lines,
            filter_code,
            clean_and_unify,
            redact_code,
        ]
    )

    orig_dataset_text = DocumentDataset.read_json(text_files, add_filename=True)
    orig_dataset_code = DocumentDataset.read_json(code_files, add_filename=True)

    # Create a histogram for different file types -text
    plot_data(orig_dataset_text, "file_size_histogram_txt.png")

    # Create a histogram for different file types - code
    plot_data(orig_dataset_code, "file_size_histogram_code.png")

    dataset_text = curation_steps_text(orig_dataset_text)
    dataset_text = dataset_text.persist()

    print(f"Original dataset length for text files: {len(orig_dataset_text.df)}")
    print(f"After dataprep: {len(dataset_text.df)}")

    dataset_code = curation_steps_code(orig_dataset_code)
    dataset_code = dataset_code.persist()

    print(f"Original dataset length for code files: {len(orig_dataset_code.df)}")
    print(f"After dataprep: {len(dataset_code.df)}")

    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    dataset_text.to_json(out_path, write_to_filename=True)
    dataset_code.to_json(out_path, write_to_filename=True)

    # Split the dataset by file category and save curated files (optional - to create blended datasets)
    separated_data_text = separate_by_metadata(
        dataset_text.df, out_path, "category"
    ).compute()
    separated_data_code = separate_by_metadata(
        dataset_code.df, out_path, "category"
    ).compute()

    client.close()


def blend_and_shuffle(
    args: Any, dataset_paths: list, dataset_weights: list, target_size: int
) -> None:
    """
    Blend and shuffle curated data based on file paths for continued pre-training

    Args:
        args (Any): Command-line arguments.
        dataset_paths (list): List containing directory paths where the different JSONL files are stored.
        dataset_weights (list): List setting weights for each directory path
        target_size (int): Target number of data samples after blending
    """
    root_path = os.path.join(DATA_DIR, "curated")
    output_path = root_path + "/data_blended"
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Blend the datasets
    datasets = [DocumentDataset.read_json(path) for path in dataset_paths]
    blended_dataset = nc.blend_datasets(target_size, datasets, dataset_weights)

    shuffle = nc.Shuffle(seed=42)
    blended_dataset = shuffle(blended_dataset)

    # Save the blend
    blended_dataset.to_json(output_path)


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 4)
    print("Args: ", args)

    # Download all the sources and get the list of text and code files.
    text_files, code_files = download_sources(100, 100, 100)
    run_curation_pipeline(args, text_files, code_files)

    # blend and shuffle datasets
    root_path = os.path.join(DATA_DIR, "curated")
    dataset_paths = [
        root_path + "/CPP",
        root_path + "/VerilogVHDL",
        root_path + "/text",
        root_path + "/Python",
    ]
    dataset_weights = [1.0, 4.0, 4.0, 1.0]
    target_size = 20
    blend_and_shuffle(args, dataset_paths, dataset_weights, target_size)


if __name__ == "__main__":
    main()
