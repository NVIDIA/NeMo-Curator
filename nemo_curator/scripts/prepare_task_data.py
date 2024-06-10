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
import pickle

import yaml

import nemo_curator
from nemo_curator.tasks.downstream_task import import_task
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    client = get_client(**ArgumentHelper.parse_client_args(args))
    # Read in config file
    with open(args.task_config_file, "r") as config_file:
        task_params = yaml.load(config_file, Loader=yaml.FullLoader)

    # Generate n-grams for all tasks
    task_list = []
    for task in task_params["tasks"]:
        print(f"Generating N-grams for task {task['name']}")
        task_class = import_task(task["name"])
        task_object = task_class(**task["params"])
        task_list.append(task_object)

    decontaminator = nemo_curator.TaskDecontamination(task_list)
    all_ngrams = decontaminator.prepare_task_ngram_count()

    with open(args.output_task_ngrams, "wb") as fp:
        pickle.dump(all_ngrams, fp)


def attach_args(
    parser=argparse.ArgumentParser(
        """
    Computes N-grams from input downstream task validation datasets.
    Takes in an input configuration file (defaults can be found under the
    config directory under the root directory of the repository) and
    writes out the computed N-grams to a pickle file where they can be
    used by the program find_matching_ngrams which will search for
    matching N-grams in the input training dataset.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    argumentHelper = ArgumentHelper(parser)
    argumentHelper.add_prepare_task_data_args()

    return argumentHelper.add_distributed_args()


def console_script():
    main(attach_args().parse_args())
