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

from nemo_curator import ConnectedComponents
from nemo_curator.cache import initialize_cache_directory
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    """
    Takes a dataset consisting of document pairs
    and their corresponding Jaccard similarity scores to compute connected
    components of documents across pairs, to find similar documents
    after applying a given threshold. The result is a dataset
    consisting of all documents that are similar (above the threshold)
    and the component they belong to.
    """
    st = time.time()
    output_path = os.path.join(args.output_dir, "connected_components.parquet")
    args.enable_spilling = True
    client = get_client(**ArgumentHelper.parse_client_args(args))

    initialize_cache_directory(args.cache_dir)

    components_stage = ConnectedComponents(
        id_column=args.input_json_id_field,
        jaccard_threshold=args.jaccard_threshold,
        false_positive_check=args.false_positive_check,
        logger=args.log_dir,
        profile_dir=args.profile_path,
    )
    components_stage.cc_workflow(output_path=output_path)
    print(f"All done in {time.time()-st:.1f} seconds")
    print(f"Results written to {output_path}")


def attach_args():
    description = "Computes connected components."
    parser = argparse.ArgumentParser(
        description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.parse_gpu_dedup_args()

    argumentHelper.add_arg_output_dir()
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
        help="Jaccard threshold below which we do not consider documents"
        " to be duplicates.",
    )
    parser.add_argument(
        "--false-positive-check",
        type=bool,
        help="Whether or not the false positive check was run before "
        "the connected components step.",
    )

    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
