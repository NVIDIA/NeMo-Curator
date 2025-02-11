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
import importlib
import os
import shutil
import time
from typing import Any, List

import numpy as np
from dask.diagnostics import ProgressBar
from dask.distributed import progress
from retriever_evalset_generator import RetrieverEvalSetGenerator

from config.config import RetrieverEvalSDGConfig
from nemo_curator import AsyncOpenAIClient, ScoreFilter, Sequential, get_client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    AnswerabilityFilter,
    EasinessFilter,
    NonAlphaNumericFilter,
)
from nemo_curator.modules.filter import Score, ScoreFilter
from nemo_curator.utils.file_utils import get_all_files_paths_under


def get_pipeline(args: Any) -> Any:

    cfg = RetrieverEvalSDGConfig.from_yaml(args.pipeline_config)
    # update api_key from input args
    cfg.api_key = args.api_key

    sdg_pipeline = Sequential(
        [
            RetrieverEvalSetGenerator(cfg),
        ]
    )
   
    return sdg_pipeline


def write_to_beir(args: Any, dataset: DocumentDataset, filtered: bool = False):

    df = dataset.df
    df = df.compute()
    if filtered:
        save_dir = os.path.join(args.output_dir, "beir", "filtered")
        qrels_save_dir = os.path.join(args.output_dir, "beir", "filtered", "qrels")
        corpus_save_path = os.path.join(
            args.output_dir, "beir", "filtered", "corpus.jsonl"
        )
        queries_save_path = os.path.join(
            args.output_dir, "beir", "filtered", "queries.jsonl"
        )
    else:
        save_dir = os.path.join(args.output_dir, "beir", "all")
        qrels_save_dir = os.path.join(args.output_dir, "beir", "all", "qrels")
        corpus_save_path = os.path.join(args.output_dir, "beir", "all", "corpus.jsonl")
        queries_save_path = os.path.join(
            args.output_dir, "beir", "all", "queries.jsonl"
        )

    os.makedirs(save_dir)
    os.makedirs(qrels_save_dir)

    df[["question-id", "question"]].rename(
        columns={"question-id": "_id", "question": "text"}
    ).to_json(queries_save_path, lines=True, orient="records")

    if filtered:
        corpus_file_path = os.path.join(args.output_dir, "beir", "all", "corpus.jsonl")
        if os.path.exists(corpus_file_path):
            shutil.copy(corpus_file_path, corpus_save_path)
        else:
            raise ValueError("Generate data first")
    else:
        df[["_id", "text"]].to_json(corpus_save_path, lines=True, orient="records")

    df[["question-id", "_id", "score"]].rename(
        columns={"question-id": "query-id", "_id": "corpus-id"}
    ).to_csv(os.path.join(qrels_save_dir, "test.tsv"), sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Input directory containing jsonl files that have the document/text chunks for query & answer generation ",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="jsonl",
        help="The input files must be in jsonl format.",
    )
    parser.add_argument(
        "--pipeline-config",
        type=str,
        default="",
        help="Pipeline configuartion yaml file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output dir for generated data",
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
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of partitions for parallel processing of data.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif not any(os.scandir(args.output_dir)):
        print("Provided directory exists but is empty, using the empty directory")
    else:
        raise ValueError("Output directory exists already, use a new directory!")

    if args.input_format == "jsonl":
        input_files = get_all_files_paths_under(args.input_dir,
                                                keep_extensions='jsonl')
        input_dataset = DocumentDataset.read_json(input_files)
    else:
        raise ValueError("Error: Only jsonl format supported")

    if args.n_partitions:
        ddf = input_dataset.df
        n_data = len(ddf)
        if args.n_partitions < n_data:
            ddf = ddf.repartition(npartitions=args.n_partitions)
            input_dataset = DocumentDataset(ddf)
        else:
            print("Number of partitions greater than data size, using 1 partition")

    sdg_pipeline = get_pipeline(args)

    print("Generating data ...")
    st_time = time.time()
    generated_dataset = sdg_pipeline(input_dataset)
    generated_dataset.persist()

    print("Writing all generated data to disk ...")
    # saving in jsonl format
    all_save_dir = os.path.join(args.output_dir, "jsonl", "all")
    os.makedirs(all_save_dir)
    generated_dataset.to_json(all_save_dir)
    generated_dataset = DocumentDataset.read_json(all_save_dir)
    print("Time taken to generate data = {:.2f} s".format(time.time() - st_time))

    # saving in beir format
    print("Write all data in beir format")
    write_to_beir(args, generated_dataset, filtered=False)

    print("RUN complete!")


if __name__ == "__main__":
    dask_client = get_client()
    main()
    # dask_client.cancel(dask_client.futures, force=True)
