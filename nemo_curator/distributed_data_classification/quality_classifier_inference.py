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
    include_model_name=False,
):
    """
    This function runs quality classification on a subset of the data.
    It loads the CFG, a data iterator, and then calls the `process_all_batches` function,
    which loads the quality classifier and runs inference.
    It also contains some additional logic to handle binary versus multiclass classification.

    Args:
        df: A Dask DataFrame partition with a "text" column and a dummy "quality_pred" column.
        max_chars: The maximum number of characters allowed in the truncated text.
        batch_size: How many samples per batch to load with PyTorch DataLoader.
        num_workers: How many subprocesses to use for PyTorch DataLoader.
        model_file_name: The path to the model file.
        labels: The list of domain labels.
        autocast: A boolean representing whether to perform inference with mixed precision.
        include_model_name: A boolean representing whether to include the model name in the "quality_pred" column name.
    Returns:
        The input Dask DataFrame with the calculated "quality_pred" column.

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
    if len(labels) == 1:
        raise ValueError("Labels must be more than 1")

    # binary case
    if len(labels) == 2:
        out_dim = 1
        binary_classification = True
    else:
        out_dim = len(labels)
        binary_classification = False

    load_model_kwargs = {
        "cfg": cfg,
        "device": device,
        "model_path": model_file_name,
        "out_dim": out_dim,
    }
    run_inference_kwargs = {
        "autocast": autocast,
        "binary_classification": binary_classification,
    }
    st = time.time()
    probs = process_all_batches(
        loader_valid,
        load_model,
        load_model_kwargs,
        run_inference,
        run_inference_kwargs,
    )
    if binary_classification:
        preds = (probs > 0.5).to(torch.int64).squeeze()
    else:
        preds = torch.argmax(probs, dim=1)
    # TODO: Do this without a CPU roundtrip in the future
    if include_model_name:
        filename = os.path.basename(model_file_name)
        df[f"quality_pred_{filename}"] = [
            labels[i] for i in preds.to("cpu").numpy().tolist()
        ]
        df[f"quality_prob_{filename}"] = probs.to("cpu").numpy().tolist()
    else:
        df["quality_pred"] = [labels[i] for i in preds.to("cpu").numpy().tolist()]
        df["quality_prob"] = probs.to("cpu").numpy().tolist()
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
    This function loads the CFG needed for quality classification.

    Returns:
        A CFG with a set `tokenizer` attribute.

    """
    cfg = CFG(max_len=1024)
    tokenizer = DebertaV2TokenizerFast.from_pretrained(cfg.model)
    cfg.tokenizer = tokenizer
    return cfg


def load_model(cfg, device, model_path, out_dim):
    """
    This function loads the quality model and prepares it to be used for inference.
    It is needed as an input to the `process_all_batches` function within the `inference_per_partition` function.

    Args:
        cfg: A CFG object.
        device: A specified PyTorch device, such as torch.device("cuda") or torch.device("cpu").
        model_path: The path to the model file.
        out_dim: An integer which corresponds to the number of labels. Use 1 for binary classification.
    Returns:
        The loaded model.

    """
    model = CustomModel(cfg, out_dim=out_dim, config_path=None, pretrained=True)
    model = model.to(device)
    sd = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def run_inference(batch, model, autocast=False, binary_classification=False):
    """
    This function runs the quality classifier on a batch of data.
    It is needed as an input to the `process_all_batches` function within the `inference_per_partition` function.

    Args:
        batch: A subset of the data as we are iterating through PyTorch DataLoader.
        model: The loaded quality classification model.
        autocast: A boolean representing whether to perform inference with mixed precision.
        binary_classification: A boolean representing whether it is a binary classification model.
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
        if binary_classification:
            probs = torch.sigmoid(out)
        else:
            probs = torch.softmax(out, dim=1)
    return probs


def add_quality_model_specific_args(parser):
    """
    This function adds a command line argument for the number of labels.

    Args:
        parser: An argparse ArgumentParser object.
    Returns:
        An argparse ArgumentParser with 1 additional argument.

    """
    parser.add_argument("--num-labels", type=int, default=3)
    return parser


def get_labels(num_labels):
    """
    This function returns a list of quality labels, depending on how many labels the user expects.

    Args:
        num_labels: An integer representing the number of possible classification labels.
    Returns:
        A list of label names.

    """
    if num_labels == 3:
        labels = ["High", "Medium", "Low"]
    elif num_labels == 2:
        labels = ["Medium_High", "Low"]
    return labels


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
        meta_df = df._meta.copy()
        meta_df["quality_pred"] = ["low"] * len(meta_df)
        meta_df["quality_prob"] = [[0, 0, 1]] * len(meta_df)
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
        f"Total time taken for quality classifier inference: {global_et-global_st} s",
        flush=True,
    )
    client.close()


def console_script():
    main()
