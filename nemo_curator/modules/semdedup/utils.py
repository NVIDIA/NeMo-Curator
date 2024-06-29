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
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd

# Import torch after other things
import torch

from nemo_curator.utils.script_utils import add_distributed_args


def merge_args_with_config(args, config):
    for key, value in config.items():
        if not getattr(args, key, None):
            setattr(args, key, value)
    return args


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="SemDedup Arguments")
    parser = add_distributed_args(parser)
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=None,
        required=False,
        help="Input directories consisting of .jsonl files that are accessible "
        "to all nodes. This path must be accessible by all machines in the cluster",
    )
    parser.add_argument(
        "--input-json-text-field",
        type=str,
        default="text",
        help="The name of the field within each json object of the jsonl "
        "file that contains the text from which minhashes will be computed. ",
    )
    parser.add_argument(
        "--input-json-id-field",
        type=str,
        default="adlr_id",
        help="The name of the field within each json object of the jsonl "
        "file that assigns a unqiue ID to each document. "
        "Can be created by running the script "
        "'./prospector/add_id.py' which adds the field 'adlr_id' "
        "to the documents in a distributed fashion",
    )
    # Set low default RMM pool size for classifier
    # to allow pytorch to grow its memory usage
    # by default
    parser.set_defaults(rmm_pool_size="1GB")
    parser.set_defaults(device="gpu")
    parser.set_defaults(set_torch_to_use_rmm=False)
    args = parser.parse_args()
    return args


def seed_everything(seed: int = 42):
    """
    Function to set seed for random number generators for reproducibility.

    Args:
        seed: The seed value to use for random number generators. Default is 42.

    Returns:
        None
    """
    # Set seed values for various random number generators
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_size(root, idmappath):
    idmappath = f"{root}/{idmappath}"
    df_map = pd.read_csv(f"{idmappath}/id_mapping.csv")
    return df_map["max_id"].max() + 1


def get_logger(
    file_name: str = "logger.log",
    level: int = logging.INFO,
    stdout: bool = False,
) -> logging.Logger:
    """
    Initialize and configure the logger object to save log entries to a file and optionally print to stdout.

    :param file_name: The name of the log file.
    :param level: The logging level to use (default: INFO).
    :param stdout: Whether to enable printing log entries to stdout (default: False).
    :return: A configured logging.Logger instance.
    """
    logger = logging.getLogger(__name__)

    # Set the logging level
    logger.setLevel(level)

    # Remove any existing handlers from the logger
    logger.handlers = []

    # Create a file handler for the logger
    file_handler = logging.FileHandler(file_name)

    # Define the formatter for the log entries
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Optionally add a stdout handler to the logger
    if stdout:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    # Return the configured logger instance
    return logger
