# Curriculum Data Preparation Tutorial

This tutorial demonstrates how a user can process a subset the LLaMa Nemotron [Post-Training Datasets](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) using NeMo Curator. The resulting dataset can be used to train a reasoning model with a modest computing budget.

The following files are needed to run this tutorial:
- The input dataset can be downloaded from Hugging Face here: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
- Be sure to request access to the tokenizer via https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- The FastText language identification model can be downloaded from here: https://fasttext.cc/docs/en/language-identification.html

This script can be run with:

```bash
python main.py \
    --input "./data" \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --model-path "./lid.176.ftz" \
    --max-token-count 8192 \
    --output-dir "./output"
```

The above script applies basic filtering to the input dataset:
- Only take samples used for Nemotron Nano training
- Remove empty and malformed samples
- Remove non-English samples
- Remove samples longer than 8k tokens (with chat template applied)

After filtering, it sorts all samples by completion length, then "interleave" thinking ON/thinking OFF for curriculum learning.
