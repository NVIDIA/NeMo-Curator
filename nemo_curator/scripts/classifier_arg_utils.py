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

import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import warnings

warnings.filterwarnings("ignore")


def add_input_output_args(parser):
    """
    This function adds the command line arguments related to input and output files.

    Args:
        parser: An argparse ArgumentParser object.
    Returns:
        An argparse ArgumentParser with 3 additional arguments.

    """
    parser.add_argument(
        "--input_file_path",
        type=str,
        help="The path of the input files",
        required=True,
    )
    parser.add_argument(
        "--input_file_type",
        type=str,
        help="The type of the input files",
        required=True,
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="The path of the output files",
        required=True,
    )
    return parser


def add_cluster_args(parser):
    """
    This function adds the command line arguments related to Dask cluster setup.

    Args:
        parser: An argparse ArgumentParser object.
    Returns:
        An argparse ArgumentParser with 8 additional arguments.

    """
    parser.add_argument(
        "--scheduler-address",
        type=str,
        default=None,
        help="""Address to the scheduler of a created Dask cluster.
                If not provided, a single node LocalCUDACluster will be started.""",
    )
    parser.add_argument(
        "--scheduler-file",
        type=str,
        default=None,
        help="""Path to the scheduler file of a created Dask cluster.
                If not provided, a single node LocalCUDACluster will be started.""",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="ucx",
        help="""Protocol to use for Dask cluster.
                Note: This only applies to the LocalCUDACluster.
                If providing a user created cluster, refer to
                https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol""",
    )
    parser.add_argument(
        "--nvlink-only",
        action="store_true",
        help="""Start a local cluster with only NVLink enabled.
                Only applicable when protocol=ucx and no scheduler file/address is specified.""",
    )
    parser.add_argument(
        "--rmm_pool_size",
        type=str,
        help="The size of the RMM pool to be used by each worker.",
        default="14GB",
    )
    parser.add_argument(
        "--CUDA_VISIBLE_DEVICES",
        type=str,
        help="The GPUs to be used by the cluster.",
        default=None,
    )
    parser.add_argument("--enable_spilling", action="store_true")
    parser.add_argument("--set_torch_to_use_rmm", action="store_true")
    return parser


def add_model_args(parser):
    """
    This function adds the command line arguments related to the model.

    Args:
        parser: An argparse ArgumentParser object.
    Returns:
        An argparse ArgumentParser with 4 additional arguments.

    """
    # Add a mutually exclusive group for model_file_name and model_file_names
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_file_name",
        type=str,
        help="The path to the model file",
        required=False,
    )
    group.add_argument(
        "--model_file_names",
        type=str,
        nargs="*",
        help="A list of model file paths",
        required=False,
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Whether to use autocast or not",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size to be used for inference",
    )
    return parser


def create_arg_parser():
    """
    This function creates the argument parser to add the command line arguments.

    Returns:
        An argparse ArgumentParser object.

    """
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-node multi-GPU inference")
    parser = add_cluster_args(parser)
    parser = add_input_output_args(parser)
    parser = add_model_args(parser)
    return parser
