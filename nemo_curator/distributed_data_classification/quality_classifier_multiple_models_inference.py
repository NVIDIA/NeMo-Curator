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
import time

from dask.distributed import wait

from nemo_curator.distributed_data_classification.arg_utils import create_arg_parser
from nemo_curator.distributed_data_classification.quality_classifier_inference import (
    add_quality_model_specific_args,
    get_labels,
    inference_per_partition,
)
from nemo_curator.utils.distributed_utils import (
    get_client,
    offload_object_on_worker,
    read_data,
    write_to_disk,
)
from nemo_curator.utils.file_utils import get_remaining_files


def delete_model_and_tokenizer_from_workers(client):
    """
    Offloads cfg_with_tokenizer and model from all Dask client workers.

    Args:
        client: A Dask client object.

    """
    task_ls = []
    # TODO: client.run does not work anymore
    # See: https://dask.discourse.group/t/cannot-run-client-run-function-when-function-contains-get-worker-in-distributed-2023-3-2-1/1772
    # find a better alternate
    for worker in client.scheduler_info()["workers"]:
        task_ls.append(
            client.submit(
                offload_object_on_worker,
                "cfg_with_tokenizer",
                workers=[worker],
                allow_other_workers=False,
                pure=False,
            )
        )
        task_ls.append(
            client.submit(
                offload_object_on_worker,
                "model",
                workers=[worker],
                allow_other_workers=False,
                pure=False,
            )
        )
    wait(task_ls)
    for t in task_ls:
        assert t.result() == True
    del task_ls


def main():
    parser = create_arg_parser()
    parser = add_quality_model_specific_args(parser)
    args = parser.parse_args()
    labels = get_labels(args.num_labels)
    print(f"Arguments parsed = {args}", flush=True)

    max_chars = 6000
    batch_size = args.batch_size
    num_workers = 0
    client = get_client(args, cluster_type="gpu")
    client.upload_file("quality_classifier_inference.py")

    print("Starting quality classifier inference", flush=True)
    global_st = time.time()
    files_per_run = len(client.scheduler_info()["workers"]) * 2
    input_files = get_remaining_files(
        args.input_file_path, args.output_file_path, args.input_file_type
    )
    print(f"Total input files {len(input_files)}", flush=True)

    if args.input_file_type == "pickle":
        add_filename = False
    else:
        add_filename = True

    for file_batch_id, i in enumerate(range(0, len(input_files), files_per_run)):
        batch_st = time.time()
        current_batch_files = input_files[i : i + files_per_run]
        print(
            f"File Batch ID {file_batch_id}: total input files {len(current_batch_files)}",
            flush=True,
        )
        df = read_data(
            input_files=current_batch_files,
            file_type=args.input_file_type,
            add_filename=add_filename,
        )
        print(f"Total input Dask DataFrame partitions {df.npartitions}", flush=True)

        for model_file_path in args.model_file_names:
            meta_df = df._meta.copy()
            model_file_name = os.path.basename(model_file_path)
            print(f"model_file_name={model_file_name}", flush=True)
            print("--" * 30, flush=True)
            meta_df[f"quality_pred_{model_file_name}"] = ["low"] * len(meta_df)
            meta_df[f"quality_prob_{model_file_name}"] = [[0, 0, 1]] * len(meta_df)
            df = df.map_partitions(
                inference_per_partition,
                max_chars,
                batch_size,
                num_workers,
                model_file_path,
                labels,
                args.autocast,
                include_model_name=True,
                meta=meta_df,
                enforce_metadata=False,
            )
            df = df.persist()
            wait(df)
            delete_model_and_tokenizer_from_workers(client)

        write_to_disk(
            df=df,
            output_file_dir=args.output_file_path,
            write_to_filename=add_filename,
        )
        batch_et = time.time()
        print(
            f"File Batch ID {file_batch_id}: completed in {batch_et-batch_st} seconds",
            flush=True,
        )

    global_et = time.time()
    print(
        f"Total time taken for multiple quality classifier inference models: {global_et-global_st} s",
        flush=True,
    )
    client.close()


def console_script():
    main()
