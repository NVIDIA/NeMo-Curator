import argparse
import json
import os
import random
from typing import Any, List

import dask
from docbuilder import LawQADownloader, LawQAIterator
from filters import FilterLowScores
from modifiers import CleanHTML
from openai import AsyncOpenAI
from synthetic_gen import SyntheticGenerator

from nemo_curator import AsyncOpenAIClient, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import WordCountFilter
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.modify import Modify
from nemo_curator.modules.semantic_dedup import SemDedup
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.script_utils import ArgumentHelper






def main():
    parser = argparse.ArgumentParser()
    parser = ArgumentHelper(parser).add_distributed_args()
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="File path of input file containing document chunks for synthetic data generation",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="rawdoc",  
        help="The synthetic data generation framework supports two input formats rawdoc or squad.",
    )
    parser.add_argument(
        "--synth-gen-rounds",
        type=int,
        default=1,
        help="How many rounds of synthetic data generation to run. Will be ignored if --synth-gen-ratio is 0.",
    )
    parser.add_argument(
        "--synth-n-variants",
        type=int,
        default=1,
        help="The number of synthetic variants to generate for each record.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="The API key to use for the synthetic data generation LLM client.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=120,
        help="The timeout value for API calls in seconds.",
    )

    args = parser.parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)
    # Don't use RMM to prevent models from running out of memory.
    args.set_torch_to_use_rmm = False

    # Prepare the download and JSONL directories.
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    train_fp, val_fp, test_fp = download_and_convert_to_jsonl()
    train_fp_curated = run_pipeline(args, train_fp)

    curated_dir = os.path.dirname(train_fp_curated)
    os.system(f"cp {val_fp} {curated_dir}")
    os.system(f"cp {test_fp} {curated_dir}")
    print(
        "--------------------------------------------------------------------------------"
    )
    print(f"Curated files are saved in '{curated_dir}'.")


if __name__ == "__main__":
    main()