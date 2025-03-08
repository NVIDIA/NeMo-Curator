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
import time

import cudf
import dask_cudf
import numpy as np

from nemo_curator import LSH
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import convert_str_id_to_int
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports():
    import cudf  # noqa: F401


def main(args):
    logger = create_logger(
        rank=0, log_file=os.path.join(args.log_dir, "rank_000.log"), name="lsh_log"
    )
    logger.info(f"Starting workflow with args:\n {args}")

    assert args.device == "gpu"
    client = get_client(**ArgumentHelper.parse_client_args(args))
    logger.info(f"Client Created {client}")
    client.run(pre_imports)
    logger.info("Pre imports complete")

    data_paths = args.input_data_dirs
    id_field = args.id_field
    minhash_field = args.input_minhash_field

    dfs = []
    for data_path in data_paths:
        dfs.append(
            dask_cudf.read_parquet(data_path, blocksize="2GB", aggregate_files=True)
        )
    df = dask_cudf.concat(dfs, ignore_unknown_divisions=True)
    df = df[~df[id_field].isna()]
    df = df.map_partitions(
        convert_str_id_to_int,
        id_field=id_field,
        meta=cudf.DataFrame(
            {minhash_field: [[1, 2, 3]], "doc_id": [1], "dataset_id": np.uint32(1)}
        ),
    )

    lsh = LSH(
        cache_dir=args.output_bucket_dir,
        num_hashes=args.minhash_length,
        num_buckets=args.num_bands,
        buckets_per_shuffle=args.buckets_per_shuffle,
        id_fields=["dataset_id", "doc_id"],
        profile_dir=args.profile_path,
        minhash_field=minhash_field,
        false_positive_check=args.false_positive_check,
        logger=logger,
    )

    t1 = time.time()
    _ = lsh(DocumentDataset(df))
    logger.info(f"Computing and writing buckets took {time.time() - t1} s")


def attach_args():
    description = """
Computes buckets from existing minhashes and writes the output
to files. Each row corresponds to a document ID, followed by the columns
denoting the bucket IDs to which the document belongs.
    """
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.parse_gpu_dedup_args()
    argumentHelper.add_arg_minhash_length()
    parser.add_argument(
        "--buckets-per-shuffle",
        type=int,
        required=True,
        help="Number of buckets to shuffle per batch.",
    )
    parser.add_argument(
        "--input-minhash-field",
        type=str,
        default="_minhash_signature",
        help="Name of the column containing minhashes.",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=20,
        help="The number of minhashes to compute for each document.",
    )
    parser.add_argument(
        "--output-bucket-dir",
        type=str,
        required=True,
        help="Output directory where minhashes will be written. "
        "Each Parquet file consists of document and bucket IDs.",
    )
    parser.add_argument(
        "--false-positive-check",
        action="store_true",
        help="Converts LSH buckets to integers required for running the false positive check",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
