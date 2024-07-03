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

import logging
import os
import time

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup import EmbeddingCreator
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir, get_remaining_files
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    semdedup_config = SemDedupConfig.from_yaml(args.config_file)
    client = get_client(**ArgumentHelper.parse_client_args(args))
    expand_outdir_and_mkdir(semdedup_config.cache_dir)
    logger = create_logger(
        rank=0,
        name="logger-compute-embeddings",
        log_file=os.path.join(semdedup_config.cache_dir, "compute_embeddings.log"),
        log_level=logging.INFO,
        stdout=True,
    )

    output_data_dir = os.path.join(
        semdedup_config.cache_dir, semdedup_config.embeddings_save_loc
    )
    # Some time jsonl files are stored as .json
    # So to handle that case we can pass the input_file_extension
    if args.input_file_extension is not None:
        input_file_extension = args.input_file_extension
    else:
        input_file_extension = args.input_file_type
    print("input_file_extension", input_file_extension)
    st = time.time()
    input_files = get_remaining_files(
        input_file_path=args.input_data_dir,
        output_file_path=output_data_dir,
        input_file_type=input_file_extension,
        num_files=semdedup_config.num_files,
    )
    logger.info(f"Processing {len(input_files)} files")
    if len(input_files) == 0:
        logger.info("No files to process")
        return

    ddf = read_data(
        input_files=input_files, file_type=args.input_file_type, add_filename=False
    )
    ddf = ddf.reset_index(drop=True)
    dataset = DocumentDataset(ddf)
    # Can repartition here if needed
    # ddf = ddf.repartition(partition_size="64MB")
    embedding_creator = EmbeddingCreator(
        model_name_or_path=semdedup_config.embedding_model_name_or_path,
        max_memory=semdedup_config.embedding_max_mem_gb,
        batch_size=semdedup_config.embedding_batch_size,
        embedding_output_dir=os.path.join(
            semdedup_config.cache_dir, semdedup_config.embeddings_save_loc
        ),
        input_column=semdedup_config.input_column,
        logger=logger,
        write_to_filename=False,
    )
    embedding_dataset = embedding_creator(dataset=dataset)
    print(embedding_dataset.df.head())
    logger.info(f"Time taken: {time.time() - st}")
    client.cancel(client.futures, force=True)
    client.close()


def attach_args():
    parser = ArgumentHelper.parse_semdedup_args(add_input_args=True)
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
