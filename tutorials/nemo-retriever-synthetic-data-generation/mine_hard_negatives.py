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
import glob
import importlib
import os
import pdb
import shutil
import time
from pathlib import Path
from typing import Any, List

from retriever_hardnegative_miner import HardNegativeMiner
from tqdm.dask import TqdmCallback

from config.config import RetrieverHardNegativeMiningConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Input dir path containing annotated data files in jsonl format",
    )
    parser.add_argument(
        "--hard-negative-mining-config",
        type=str,
        default="",
        help="Configuration yaml file path containing config for hard negative mining",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output file containing hard negatives",
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

    if not os.path.exists(args.input_dir):
        raise ValueError("Input directory not found")

    if os.path.exists(args.output_dir):
        raise ValueError("Output dir exists already, use a new file name!")

    if args.input_dir:
        input_dataset = DocumentDataset.read_json(args.input_dir)
        # input_dataset = DocumentDataset.read_json(os.path.join(args.input_dir,"clustered_dataset"))
    else:
        raise ValueError("provide input file path")

    if args.hard_negative_mining_config:
        cfg = RetrieverHardNegativeMiningConfig.from_yaml(
            args.hard_negative_mining_config
        )

    else:
        raise ValueError("provide config for hard negative mining")
    if args.api_key:
        cfg.api_key = args.api_key

    mine_hard_negatives = HardNegativeMiner(cfg)
    print("Mining hard negatives ...")
    st_time = time.time()
    mined_dataset = mine_hard_negatives(input_dataset)

    print("Time taken = {:.2f} s".format(time.time() - st_time))
    print("Saving data in jsonl format ...")
    mined_dataset.df.to_json(
        os.path.join(args.output_dir, "mined_dataset"), lines=True, orient="records"
    )


if __name__ == "__main__":
    dask_client = get_client(cluster_type="cpu")
    main()
