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
import random

import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
)
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    client = get_client(**ArgumentHelper.parse_client_args(args))

    output_dir = expand_outdir_and_mkdir(args.output_data_dir)
    files = get_all_files_paths_under(args.input_data_dir)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(files)

    dataset = DocumentDataset(
        read_data(
            files, file_type=args.input_file_type, backend="pandas", add_filename=True
        )
    )
    add_id = nemo_curator.AddId(
        args.id_field_name, id_prefix=args.id_prefix, start_index=args.starting_index
    )
    id_dataset = add_id(dataset)

    write_to_disk(
        id_dataset.df,
        output_dir,
        write_to_filename=True,
        output_type=args.output_file_type,
    )


def attach_args(
    parser=argparse.ArgumentParser(
        """
Adds unique identifiers to each document in the dataset.
Creates a new ID field with name specified by the argument
"--id-field-name" within each json.

This script essentially works by counting the total
number of documents within the dataset and then, in parallel
assigns unique sequential ids to each document in the dataset.

If a document identifier does not already exist for each document, then
these ids must be added prior to performing fuzzy/exact deduplication
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    argumentHelper = ArgumentHelper(parser)

    argumentHelper.add_id_field_name()
    argumentHelper.add_id_prefix()
    argumentHelper.add_input_data_dir()
    argumentHelper.add_input_file_type()
    argumentHelper.add_output_data_dir(
        help="The output directory to where the jsonl files with ids will "
        "be written. If not specified, the ids will be written in-place"
    )
    argumentHelper.add_output_file_type()
    argumentHelper.add_seed()
    argumentHelper.add_starting_index()
    argumentHelper.add_shufle(
        help="Shuffle the order of files before assigning IDs."
        "Useful for creating a copy dataset with different IDs"
    )

    return argumentHelper.add_distributed_args()


def console_script():
    main(attach_args().parse_args())
