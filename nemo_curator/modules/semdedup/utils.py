import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd

# Import torch after other things
import torch
import yaml

from nemo_curator.utils.script_utils import add_distributed_args


def merge_args_with_config(args, config):
    for key, value in config.items():
        if not getattr(args, key, None):
            print(f"Setting {key} to {value}")
            setattr(args, key, value)
    return args


def parse_arguments():
    parser = ArgumentParser(description="SemDedup Arguments")
    parser = add_distributed_args(parser)
    # Set low default RMM pool size for classifier
    # to allow pytorch to grow its memory usage
    # by default
    parser.set_defaults(rmm_pool_size="4GB")
    parser.set_defaults(device="gpu")
    parser.set_defaults(set_torch_to_use_rmm=False)
    parser.add_argument(
        "--config_file", help="YAML with configs", default="config.yaml"
    )
    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as y_file:
        config_args = yaml.load(y_file, Loader=yaml.FullLoader)

    return merge_args_with_config(args, config_args)


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
