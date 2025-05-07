# Curriculum Data Preparation Tutorial

This tutorial demonstrates how a user can process a subset the LLaMa Nemotron [Post-Training Datasets](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) using NeMo Curator. The resulting dataset can be used to train a reasoning model with a modest computing budget.

The following files are needed to run this tutorial:
- The input dataset can be downloaded from Hugging Face here: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
- Be sure to request access to the tokenizer via https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- The FastText language identification model can be downloaded from here: https://fasttext.cc/docs/en/language-identification.html

The following command can be used to download the FastText language identification model, which is used in the next step as the `--lang-id-model-path`:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

This script can be run with:

```bash
python main.py \
    --input "./data" \
    --filename-filter "code" "math" \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --lang-id-model-path "./lid.176.ftz" \
    --max-token-count 8192 \
    --output-dir "./output" \
    --device "gpu"
```

By default, the JSONL data is read with a blocksize of 1 GB per Dask partition. To customize the file reading logic, the user may specify `--json-blocksize "1gb"` with any string representation for the partition size (e.g., "1gb", "256mb"). ALternatively, the user may specify `--json-files-per-partition 2` with any integer to represent the number of JSONL files per Dask partition. Please note that either the blocksize or files per partition can be specified, but not both.

To filter files by name, the user may pass `--filename-filter` followed by any number of strings, such as "code" and "math". When reading the input data directory, the list of files will be filtered to only include files with names containing at least 1 of the strings provided by `--filename-filter`. If `--filename-filter` is not specified, then all files within the directory will be used.

The above script applies basic filtering to the input dataset:
- Only take samples used for Nemotron Nano training
- Remove empty and malformed samples
- Remove non-English samples
- Remove samples longer than 8k tokens (with chat template applied via the tokenizer). For faster performance, the user can specify the boolean flag `--skip-tokenize`, which skips the tokenization step and returns the text length (number of characters) instead.

After filtering, it sorts all samples by completion length, then "interleaves" thinking ON/thinking OFF for curriculum learning. For large datasets, we recommend setting `--device "gpu"` and using an approximate interleaving approach which interleaves the data by Dask partition rather than row by row. If you prefer to interleave globally (per row), then you can specify the boolean flag `--global-interleave` from the script command.
