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

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd
from utils import (
    TextLineCountFilter,
    clean_and_unify,
    exact_dedupe,
    filter_text,
    fuzzy_dedupe,
    redact_PII,
    rm_dir,
    semantic_dedupe,
)

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import separate_by_metadata
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
CONFIG_DIR = os.path.join(SCRIPT_DIR_PATH, "configs")


def process_textual_data(textual_data):
    textual_data_list = []
    for text_data in textual_data:
        encodings = "None"
        text = text_data["metadata"]["content"]
        category = text_data["document_type"]
        line_count = len(text.split("\n"))
        id = text_data["metadata"]["source_metadata"]["source_id"]
        file_path = text_data["metadata"]["source_metadata"]["source_name"]
        filename = Path(file_path).stem
        file_extension = Path(file_path).suffix

        textual_data_list.append(
            [
                encodings,
                text,
                id,
                file_extension,
                filename,
                category,
                line_count,
                file_path,
            ]
        )

    return pd.DataFrame(
        textual_data_list,
        columns=[
            "encodings",
            "text",
            "id",
            "file_extension",
            "file_name",
            "category",
            "line_count",
            "path",
        ],
    )


def process_image_data(image_data):
    image_data_list = []
    for img_data in image_data:
        encodings = img_data["metadata"]["content"]
        text = img_data["metadata"]["image_metadata"]["caption"]
        category = img_data["document_type"]
        line_count = 0
        id = img_data["metadata"]["source_metadata"]["source_id"]
        file_path = img_data["metadata"]["source_metadata"]["source_name"]
        filename = Path(file_path).stem
        file_extension = Path(file_path).suffix

        image_data_list.append(
            [
                encodings,
                text,
                id,
                file_extension,
                filename,
                category,
                line_count,
                file_path,
            ]
        )

    return pd.DataFrame(
        image_data_list,
        columns=[
            "encodings",
            "text",
            "id",
            "file_extension",
            "file_name",
            "category",
            "line_count",
            "path",
        ],
    )


def process_structured_data(structured_data):
    struct_data_list = []
    for struct_data in structured_data:
        encodings = struct_data["metadata"]["content"]
        text = struct_data["metadata"]["table_metadata"]["table_content"]
        category = struct_data["document_type"]
        line_count = len(text.split("\n"))
        id = struct_data["metadata"]["source_metadata"]["source_id"]
        file_path = struct_data["metadata"]["source_metadata"]["source_name"]
        filename = Path(file_path).stem
        file_extension = Path(file_path).suffix

        struct_data_list.append(
            [
                encodings,
                text,
                id,
                file_extension,
                filename,
                category,
                line_count,
                file_path,
            ]
        )

    return pd.DataFrame(
        struct_data_list,
        columns=[
            "encodings",
            "text",
            "id",
            "file_extension",
            "file_name",
            "category",
            "line_count",
            "path",
        ],
    )


def process_data(data_type_map):
    text_df = process_textual_data(data_type_map["text"])
    image_df = process_image_data(data_type_map["image"])
    struct_df = process_structured_data(data_type_map["structured"])

    text_ddf = DocumentDataset(dd.from_pandas(text_df, npartitions=2))
    image_ddf = DocumentDataset(dd.from_pandas(image_df, npartitions=2))
    struct_ddf = DocumentDataset(dd.from_pandas(struct_df, npartitions=2))

    return text_ddf, image_ddf, struct_ddf


def run_text_curation_pipeline(args: Any, text_ddf, struct_ddf) -> None:
    """
    Run the curation pipeline on the Wiki+Arxiv+Github datasets.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    # Initialize the Dask cluster.
    client = get_client(
        **ArgumentHelper.parse_client_args(args), set_torch_to_use_rmm=True
    )

    # Define data curation steps for text and pdf files
    curation_steps_text = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                TextLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_text,
            exact_dedupe,
            redact_PII,
        ]
    )

    # create a field combining fields file type and line count
    text_ddf.df["file_type_count"] = (
        text_ddf.df["category"] + " : " + text_ddf.df["line_count"].astype(str)
    )

    struct_ddf.df["file_type_count"] = (
        struct_ddf.df["category"] + " : " + struct_ddf.df["line_count"].astype(str)
    )

    print("Executing the curation pipeline...")
    dataset_text = curation_steps_text(text_ddf)
    dataset_struct = curation_steps_text(struct_ddf)

    datasets = [dataset_text, dataset_struct]
    counts = {"person_count": 0, "email_count": 0}

    for dataset in datasets:
        counts["person_count"] += (
            dataset.df["text"].str.count(r"\bPERSON\b").sum().compute()
        )
        counts["email_count"] += (
            dataset.df["text"].str.count(r"\bEMAIL_ADDRESS\b").sum().compute()
        )

    person_count, email_count = counts["person_count"], counts["email_count"]

    print(f"Original dataset length for text: {len(text_ddf.df)}")
    print(f"Original dataset length for tables and charts: {len(struct_ddf.df)}")

    print(f"After preprocessing text data: {len(dataset_text.df)}")
    print(f"After preprocessing tables and charts: {len(dataset_struct.df)}")

    print(
        f"Redacted names and email address for text, tables and charts: {person_count, email_count}"
    )

    if args.device == "gpu":
        print("Executing the semantic dedupe pipeline...")
        gpu_dataset_text = DocumentDataset(dataset_text.df.to_backend("cudf"))
        gpu_dataset_struct = DocumentDataset(dataset_struct.df.to_backend("cudf"))

        text_sem_dedupe_config_yaml_path = os.path.join(
            CONFIG_DIR, "text_semantic_dedupe_config.yaml"
        )
        struct_sem_dedupe_config_yaml_path = os.path.join(
            CONFIG_DIR, "struct_semantic_dedupe_config.yaml"
        )

        cache_dir_txt = os.path.join(SCRIPT_DIR_PATH, "cache", "semdedup_cache", "text")
        cache_dir_struct = os.path.join(
            SCRIPT_DIR_PATH, "cache", "semdedup_cache", "struct"
        )

        rm_dir(cache_dir_txt)
        rm_dir(cache_dir_struct)

        text_duplicates = semantic_dedupe(
            dataset=gpu_dataset_text,
            sem_dedupe_config_yaml_path=text_sem_dedupe_config_yaml_path,
        )
        struct_duplicates = semantic_dedupe(
            dataset=gpu_dataset_struct,
            sem_dedupe_config_yaml_path=struct_sem_dedupe_config_yaml_path,
        )

        text_unique_ids = text_duplicates.df.to_backend("pandas").compute()["id"]
        struct_unique_ids = struct_duplicates.df.to_backend("pandas").compute()["id"]

        semantic_dataset_text = DocumentDataset(
            gpu_dataset_text.df[gpu_dataset_text.df.id.isin(text_unique_ids)]
        )
        semantic_dataset_struct = DocumentDataset(
            gpu_dataset_struct.df[gpu_dataset_struct.df.id.isin(struct_unique_ids)]
        )

        print(f"After semantic dedupe for text: {len(semantic_dataset_text.df)}")
        print(
            f"After semantic dedupe for tables and charts: {len(semantic_dataset_struct.df)}"
        )

        print("Executing the fuzzy dedupe pipeline...")
        cache_dir_txt = os.path.join(SCRIPT_DIR_PATH, "cache", "fuzzy_dedupe", "text")
        cache_dir_struct = os.path.join(
            SCRIPT_DIR_PATH, "cache", "fuzzy_dedupe", "struct"
        )

        rm_dir(cache_dir_txt)
        rm_dir(cache_dir_struct)

        fuzzy_dataset_text = fuzzy_dedupe(
            dataset=semantic_dataset_text, cache=cache_dir_txt
        )
        fuzzy_dataset_struct = fuzzy_dedupe(
            dataset=semantic_dataset_struct, cache=cache_dir_struct
        )

        dataset_text.df = fuzzy_dataset_text.df.to_backend("pandas")
        dataset_struct.df = fuzzy_dataset_struct.df.to_backend("pandas")
        print(f"After fuzzy dedupe for text files: {len(dataset_text.df)}")
        print(
            f"After fuzzy dedupe for tables and charts files: {len(dataset_struct.df)}"
        )

    final_dataset_text = dataset_text.persist()
    final_dataset_struct = dataset_struct.persist()

    print("Writing the results to disk...")
    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    final_dataset_text.to_json(out_path, write_to_filename=True)
    final_dataset_struct.to_json(out_path, write_to_filename=True)

    print("Writing results to disk completed")

    # Split the dataset by file category and save curated files (optional - to create blended datasets)
    print("Split dataset by metadata")

    separated_data_text = separate_by_metadata(
        final_dataset_text.df, out_path, "category"
    ).compute()
    separated_data_struct = separate_by_metadata(
        final_dataset_struct.df, out_path, "category"
    ).compute()

    client.close()


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 2)
    args.device = "gpu"
    print("Args: ", args)

    nv_ingest_path = "../ingest/sources/separated_extracted_data/data_type_map.json"
    with open(nv_ingest_path, "r", encoding="utf-8") as f:
        data_type_map = json.load(f)

    text_ddf, image_ddf, struct_ddf = process_data(data_type_map)
    run_text_curation_pipeline(args, text_ddf, struct_ddf)
    # run_image_curation_pipeline(args, image_ddf)


if __name__ == "__main__":
    main()
