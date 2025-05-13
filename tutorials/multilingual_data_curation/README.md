# Multilingual Data Curation Pipeline with NeMo Curator

The following files demonstrate how to curate a multilingual Common Crawl dataset using NeMo Curator.
The pipeline is as follows:

- *Step 1*: Download and extract Common Crawl data. By default, the pipeline is set to extract the April 2025 Common Crawl snapshot with the jusText text extraction algorithm and its default settings.
- *Step 2*: Language identification. The [FastText language identification model](https://fasttext.cc/docs/en/language-identification.html) can be downloaded with:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

- *Step 3*: Add ID field. In order to run deduplication with NeMo Curator, text datasets must have an ID column.
- *Step 4*: Exact deduplication.
- *Step 5*: Fuzzy deduplication.
- *Step 6*: Exact substring deduplication.

Above, steps 1-3 and step 6 are run on the CPU. Meanwhile, steps 4 and 5 are run on the GPU.

To start a job, please edit `scripts/start-slurm.sh` to match the expectations and parameters of the current job.
Then, submit your job with `sbatch start-slurm.sh`.

To perform exact substring deduplication, please refer to: https://github.com/google-research/deduplicate-text-datasets/tree/master
