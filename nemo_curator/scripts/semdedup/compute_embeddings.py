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
from typing import List

from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.modules.config import SemDedupConfig
from nemo_curator.modules.semantic_dedup import EmbeddingCreator
from nemo_curator.utils.distributed_utils import get_client, read_data
from nemo_curator.utils.file_utils import get_remaining_files
from nemo_curator.utils.script_utils import ArgumentHelper


def get_input_files(
    input_data_dir: str, input_file_type: str, output_data_dir: str, num_files: int
) -> List[str]:
    os.makedirs(output_data_dir, exist_ok=True)
    len_written_files = len(os.listdir(output_data_dir))
    input_files = get_remaining_files(input_data_dir, output_data_dir, input_file_type)
    # Gaurd against non-json files present in the input directory
    input_files = [f for f in input_files if f.endswith(input_file_type)]
    if num_files > 0:
        left_to_sample = max(num_files - len_written_files, 0)
    else:
        left_to_sample = len(input_files)

    input_files = input_files[:left_to_sample]
    return input_files


def main(args):
    semdedup_config = SemDedupConfig.from_yaml(args.config_file)
    client = get_client(**ArgumentHelper.parse_client_args(args))
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
    st = time.time()
    input_files = get_input_files(
        input_data_dir=args.input_data_dir,
        input_file_type=args.input_file_type,
        output_data_dir=output_data_dir,
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
        logger=logger,
        write_to_filename=False,
    )
    embedding_dataset = embedding_creator(
        dataset=dataset, input_column=args.input_text_field
    )
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
