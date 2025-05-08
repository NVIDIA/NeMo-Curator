# Curriculum Data Preparation Tutorial

This tutorial demonstrates how a user can process a subset the LLaMa Nemotron [Post-Training Datasets](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) using NeMo Curator. The resulting dataset can be used to train a reasoning model with a modest computing budget.

## Environment Setup

Setup requirements:

- Hardware: CPU is sufficient, GPU is recommended for enhanced performance
- Recommended environment: This tutorial was developed and tested with a Conda environment

Please refer to NeMo Curator's [README](https://github.com/NVIDIA/NeMo-Curator?tab=readme-ov-file#get-started) for instructions on how to download NeMo Curator via PyPI, source, or Docker.

## Prerequisites

### Download input dataset

The input dataset can be downloaded from Hugging Face here: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset

The following commands can be used to download the dataset:

```bash
git lfs install
git clone https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
```

### Tokenizer access instructions

The tokenizer used by this tutorial is called [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Using it requires requesting access:

1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Access request"
3. Fill out the form and wait for approval
4. Once approved, log in to your Hugging Face account via the Hugging Face CLI. In the terminal, this can be done via `huggingface-cli login`

### Download FastText language identification model

The FastText language identification model is used to identify and filter out non-English text from the dataset. It can be downloaded from here: https://fasttext.cc/docs/en/language-identification.html

The following command can be used to download the FastText language identification model to your current working directory:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -P ./
```

## Usage

This tutorial can be run with:

```bash
python main.py \
    --input-dir "/path/to/input/data" \
    --filename-filter "code" "math" \
    --remove-columns "category" "generator" "license" "used_in_training" "version" \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --lang-id-model-path "./lid.176.ftz" \
    --max-token-count 16000 \
    --max-completion-token-count 8192 \
    --output-dir "/path/to/output" \
    --device "gpu" \
    --n-workers 4
```

Since the entire input dataset is very large, we recommend choosing a subset of the dataset to suit your needs. To help with this, we provide a way to filter files before reading. To filter files by name, the user may pass `--filename-filter` followed by any number of strings, such as "code" and "math". When reading the input data directory, the list of files will be filtered to only include files with names containing at least 1 of the strings provided by `--filename-filter`. If `--filename-filter` is not specified, then all files within the directory (over 30 million rows) will be used.

The above script applies basic filtering to the input dataset:

- Only take samples used for Nemotron Nano training
- Remove empty and malformed samples
- Remove non-English samples
- Remove samples with total length (system prompt, input, and output responses) longer than 16k tokens (with chat template applied via the tokenizer)
- Remove samples with output responses longer than 8k tokens (with chat template applied via the tokenizer)
- Remove any columns specified by the `--remove-columns` parameter. We recommend removing the columns specified above

After filtering, it sorts all samples by completion (output response) length, then "interleaves" thinking ON/thinking OFF for curriculum learning. For large datasets, we recommend setting `--device "gpu"` and using an approximate interleaving approach which interleaves the data by Dask partition rather than row by row. If you prefer to interleave globally (per row), then you can specify the boolean flag `--global-interleave` from the script command.

If you are interested in counting and displaying the number of rows after each step in the pipeline, then you can specify the boolean flag `--generate-statistics` from the script command. Please note that enabling this is computationally expensive and will slow down the pipeline, so it is not recommended for large datasets. The default value is False.

## Debugging Out of Memory Errors

If you are running into out of memory (OOM) errors, there are a couple of approaches you can try. One is to avoid very large partitions of data. By default, the JSONL data is read with a blocksize of 256 MB per Dask partition. To customize the file reading logic, the user may specify `--json-blocksize "1gb"` with any string representation for the partition size (e.g., "1gb", "256mb"). Alternatively, the user may specify `--json-files-per-partition 2` with any integer to represent the number of JSONL files per Dask partition. Please note that either the blocksize or files per partition can be specified, but not both. For GPU workflows, a good general rule of thumb is to set the blocksize to 1/32 of the total GPU memory. In general, a blocksize between 100 MB and 1 GB is considered ideal.

You may also encounter errors about Dask workers unexpectedly shutting down. To help mitigate this, consider lowering the `--n-workers` parameter. By default, we set the number of Dask workers equal to the number of CPU cores. It may be helpful to set `--n-workers` to half or a fourth of the number of CPU cores and possibly reduce the number from there. For example, if `lscpu` shows `CPU(s): 96`, then setting `--n-workers 48` or `--n-workers 24` may help optimize performance while avoiding memory issues. In the example bash script, we set `--n-workers 4` as a very safe option to help avoid errors.
