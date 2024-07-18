# Data curation for DAPT (Domain Adaptive Pre-Training)

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain adapted LLM. LLama models are continually pre-trained with 20B plus tokens on domain-specific chip design data, including code, documents, etc., based on NeMo foundation models and then fine-tuned with instruction datasets from design data as well as external sources.

Here, we share a tutotial with best practices on data curation for DAPT (domain-adaptive pre-training) for a ChipNeMo-like code generation use case.

* In this tutorial, we will leverage chip domain/hardware datasets from open-source GitHub repositories (`.code/sources/github_repos.jsonl`), wiki URLs (`.code/sources/wikipedia_urls.jsonl`), and academic papers(`./sources/arxiv_urls.jsonl`).

* `\code\data` contains curated data after processing

The small size of this dataset makes it ideal for creating and validating data curation pipelines on a local machine or a cluster.

This playbook utilizes specific tools and techniques. First, we convert all files to Txt format (if not already in Txt), compress files on disk, add metadata, and convert them to JSON (`./data/raw/`). Then, we leverage [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator/tree/main) to mine high-quality text at scale from a massive code-generation corpus. We use its capabilities to extract text, identify code file types, fix unicode errors, filter quality through heuristics, deduplicate, and redact personal information. We finally also provide steps to blend and shuffle data sources for continued pre-training.


## Hardware Requirements
* This playbook can run on CPUs or GPUs. For GPUs, this playbook has been tested on minimum 1xA100 80G using nvcr.io/nvidia/nemo:24.03.01.framework container


## Walkthrough

We will use the datasets in the `dapt-curation/code/data` folder to illustrate data curation through this pipeline. Specifically sample data collected in:
* `./data/raw/github` (we clone github repos, extract text from each file and convert to jsonl)
* `./data/raw/arxiv_pdfs` (we extract data from pdfs, convert to txt and store as jsonl files)
* `./data/raw/wikipedia` (we extract data from htmls, parse, convert to txt and store as json files)

The tutorial follows the steps below:<br>
- Step 1: Install requirements and import libraries<br>
- Step 2: Download the data from online sources (Github repos, wiki urls, arxiv pdfs), extract metadata and convert to JSONL<br>
- Step 3: Load the dataset <br>
- Step 4: Examine the file types and sizes (optional) <br>
- Step 5: Run the data curation pipeline with with Nemo Curator<br>
    - File type identification and separation
    - Document-level exact deduplication
    - Heuristic-based quality filtering (Number of lines, worc count, top N-grams, etc.)
    - Fix unicode errors via ftfy
    - PII redaction
- Step 6: Save the filtered and curated data <br>
- Step 8: Blend datasets and shuffle


## Usage

After installing the NeMo Curator package, install the dependencies and run:

`pip install -r requirements.txt`

`python code/main.py`

This will download chip-design related datasets and begin the data curation pipeline.


## TODO
* Add instructions for apt install tessaract-ocr
* Add improved PII redaction, classifier based filtering & fuzzy dedupe
