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
import warnings

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import torch
from packaging import version
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from nemo_curator.distributed_data_classification.arg_utils import create_arg_parser
from nemo_curator.distributed_data_classification.pytorch_utils import (
    CFG,
    CustomModel,
    TestDataset,
    collate,
)
from nemo_curator.utils.distributed_utils import (
    get_client,
    load_object_on_worker,
    process_all_batches,
    read_data,
    write_to_disk,
)
from nemo_curator.utils.file_utils import get_remaining_files

warnings.filterwarnings("ignore")


def inference_per_partition(
    df,
    max_chars,
    batch_size,
    num_workers,
    model_file_name,
    labels,
    autocast,
):
    """
    This function runs domain classification on a subset of the data.
    It loads the CFG, a data iterator, and then calls the `process_all_batches` function,
    which loads the domain classifier and runs inference.

    Args:
        df: A Dask DataFrame partition with a "text" column and a dummy "pred" column.
        max_chars: The maximum number of characters allowed in the truncated text.
        batch_size: How many samples per batch to load with PyTorch DataLoader.
        num_workers: How many subprocesses to use for PyTorch DataLoader.
        model_file_name: The path to the model file.
        labels: The list of domain labels.
        autocast: A boolean representing whether to perform inference with mixed precision.
    Returns:
        The input Dask DataFrame with the calculated "pred" column.

    """
    cfg = cfg_per_partition()

    dataset_valid = TestDataset(cfg, df, max_chars)
    loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    device = torch.device("cuda")
    load_model_kwargs = {"cfg": cfg, "device": device, "model_path": model_file_name}
    run_inference_kwargs = {"autocast": autocast}
    st = time.time()
    preds = process_all_batches(
        loader_valid,
        load_model,
        load_model_kwargs,
        run_inference,
        run_inference_kwargs,
    )
    preds = preds.cpu().numpy()
    df["pred"] = [labels[i] for i in preds]

    et = time.time()
    print(
        f"Time taken for inference for num_batches: {len(loader_valid)} : {et-st} s",
        flush=True,
    )

    return df


def cfg_per_partition():
    """
    This function loads the CFG on the worker currently running the task.
    See `load_object_on_worker` function.

    Returns:
        A CFG with a set `tokenizer` attribute.

    """
    return load_object_on_worker("cfg_with_tokenizer", load_cfg_with_tokenizer, {})


def load_cfg_with_tokenizer():
    """
    This function loads the CFG needed for domain classification.

    Returns:
        A CFG with a set `tokenizer` attribute.

    """
    cfg = CFG()
    tokenizer = DebertaV2TokenizerFast.from_pretrained(cfg.model)
    cfg.tokenizer = tokenizer
    return cfg


def load_model(cfg, device, model_path):
    """
    This function loads the domain model and prepares it to be used for inference.
    It is needed as an input to the `process_all_batches` function within the `inference_per_partition` function.

    Args:
        cfg: A CFG object.
        device: A specified PyTorch device, such as torch.device("cuda") or torch.device("cpu").
        model_path: The path to the model file.
    Returns:
        The loaded model.

    """
    model = CustomModel(cfg, out_dim=27, config_path=None, pretrained=True)
    model = model.to(device)
    sd = torch.load(os.path.join(model_path), map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
    if version.parse(TRANSFORMERS_VERSION) >= version.parse("4.31.0"):
        sd.pop("model.embeddings.position_ids", None)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def run_inference(batch, model, autocast=False):
    """
    This function runs the domain classifier on a batch of data.
    It is needed as an input to the `process_all_batches` function within the `inference_per_partition` function.

    Args:
        batch: A subset of the data as we are iterating through PyTorch DataLoader.
        model: The loaded domain classification model.
        autocast: A boolean representing whether to perform inference with mixed precision.
    Returns:
        A tensor of predictions.

    """
    with torch.no_grad():
        batch = collate(batch)
        if autocast:
            with torch.autocast(device_type="cuda"):
                out = model(batch)[:, 0, :]
        else:
            out = model(batch)[:, 0, :]
        pred_idx = torch.sigmoid(out).argmax(1)

    return pred_idx


def main():
    labels = [
        "Adult",
        "Arts_and_Entertainment",
        "Autos_and_Vehicles",
        "Beauty_and_Fitness",
        "Books_and_Literature",
        "Business_and_Industrial",
        "Computers_and_Electronics",
        "Finance",
        "Food_and_Drink",
        "Games",
        "Health",
        "Hobbies_and_Leisure",
        "Home_and_Garden",
        "Internet_and_Telecom",
        "Jobs_and_Education",
        "Law_and_Government",
        "News",
        "Online_Communities",
        "People_and_Society",
        "Pets_and_Animals",
        "Real_Estate",
        "Reference",
        "Science",
        "Sensitive_Subjects",
        "Shopping",
        "Sports",
        "Travel_and_Transportation",
    ]

    args = create_arg_parser().parse_args()
    print(f"Arguments parsed = {args}", flush=True)
    max_chars = 2000
    batch_size = args.batch_size
    num_workers = 0

    client = get_client(args, cluster_type="gpu")
    print("Starting domain classifier inference", flush=True)
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
        meta_df = df._meta.copy()
        meta_df["pred"] = [0] * len(meta_df)
        df = df.map_partitions(
            inference_per_partition,
            max_chars,
            batch_size,
            num_workers,
            args.model_file_name,
            labels,
            args.autocast,
            meta=meta_df,
            enforce_metadata=False,
        )
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
        f"Total time taken for domain classifier inference: {global_et-global_st} s",
        flush=True,
    )
    client.close()


def console_script():
    main()
