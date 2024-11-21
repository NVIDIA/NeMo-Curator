# Copyright (c) 2024, NVIDIA CORPORATION.
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
import time

os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"

import logging

from nemo_curator.classifiers import QualityClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.file_utils import get_remaining_files

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeMo quality classifier.")
    parser.add_argument("--input", help="Path to the input folder")
    parser.add_argument("--output", help="Path to the output folder")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for quality model"
    )
    args = parser.parse_args()

    t0 = time.time()
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")
    logging.info(client)

    classifier = QualityClassifier(batch_size=args.batch_size)

    raw_base_path = args.input
    qm_base_path = args.output
    files = get_remaining_files(raw_base_path, qm_base_path, "parquet")
    if files:
        logging.info(f"Found {len(files)} remaining files for processing")
        input_dataset = DocumentDataset.read_parquet(
            files, backend="cudf", add_filename=True
        )
        result_dataset = classifier(dataset=input_dataset)
        result_dataset.to_parquet(qm_base_path, write_to_filename=True)
    else:
        logging.info("Nothing to be done. All files are already processed.")

    logging.info(f"Done in {time.time() - t0}")
