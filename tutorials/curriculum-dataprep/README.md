# Curriculum Data Preparation Tutorial

This script can be run with:

```bash
python main.py \
    --input "./data" \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --model-path "./lid.176.ftz" \
    --max-token-count 8192 \
    --output-dir "./output"
```

- The input dataset can be downloaded from Hugging Face here: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset
- Be sure to request access to the tokenizer via https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- The FastText language identification model can be downloaded from here: https://fasttext.cc/docs/en/language-identification.html
