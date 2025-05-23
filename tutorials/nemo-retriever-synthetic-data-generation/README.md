# NeMo Retriever Synthetic Data Generation

NeMo Retriever Synthetic Data Generation (SDG) is designed to streamline the creation of high-quality evaluation datasets for Text QA retrieval use cases. By leveraging existing enterprise data, this pipeline enables rapid generation of relevant evaluation datasets, facilitating improved model performance.

This version supports the generation of evaluation datasets, creating synthetic benchmark datasets compatible with commonly used evaluation frameworks such as [BEIR](https://huggingface.co/datasets/BeIR/beir). Synthetic training dataset generation will be supported in an upcoming version.

NeMo Retriever SDG can be run either from the command line, or using the [notebook example](notebooks/quickstart.ipynb) provided in this repository. Check the [Prerequisites](#prerequisites) section for instructions on generating an API key and installing libraries. To get started with the notebook, follow the [Notebook Quick Start](#run-pipeline-ipython-notebook) instructions. Otherwise, follow the [CLI Quick Start](#run-pipeline-cli) section.

![NeMo Retriever SDG](figures/sdg_pipeline.png)

#### Key Features

* Quickly generate complex QA datasets from existing text documents for retriever model evaluation.
* Output datasets can be formatted in [SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/rajpurkar/squad) or [BEIR (Benchmarking Information Retrieval)](https://huggingface.co/datasets/BeIR/beir) format for easy integration into evaluation workflows.
* Designed to integrate seamlessly with [NVIDIA NeMo Evaluator](https://developer.nvidia.com/nemo-microservices) microservice, currently in early access.


## Quickstart

### Prerequisites

In order to use NeMo Retriever SDG, you will need access to NVIDIA’s API Catalog. Go to the [NGC Personal Key Manager](https://org.ngc.nvidia.com/setup) to generate a Personal Key that will allow you to access AI Foundation Models and Endpoints.

To install the required libraries, navigate to the root directory of the project and run the following command in your notebook or command line:

```
$ pip install -r requirements.txt
```

Alternatively, you can use container `nvcr.io/nvidia/nemo:24.09`.

```
$ docker pull nvcr.io/nvidia/nemo:24.09

$ docker run -it --rm --gpus all --ipc host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.09

/workspace# pip install -r requirements.txt
/workspace# jupyter notebook
```


### Run Pipeline (iPython notebook)

Navigate to the [quick start notebook](notebooks/quickstart.ipynb) and follow the instructions.

### Run Pipeline (CLI)

The pipeline can be run with datasets in ```jsonl``` (only text, title and ids if any) format. To test the pipeline, you can use the provided example data at ```sample_data/sample_data_rawdoc.jsonl```

To use jsonl format, provide your data in a single or multiple `.jsonl` files. The structure of the data should follow this format: `{"text": <document>, "title": <title>}`. Additionally, if the documents already have a document id, the input file can also contain document ids. The same ids will be persisted in the generated data as well. Another accepted format is `{"_id": <document_id>, "text": <document>, "title": <title>}`.

The pipeline can be run in two modes (1. Generation and 2. Filtering). In order to run the full pipeline in generation mode, use the script ```main.py``` with the flag ```--pipeline-type=generate```
```
python tutorials/nemo-retriever-synthetic-data-generation/main.py \
  --api-key=<API Key> \
  --input-dir=tutorials/nemo-retriever-synthetic-data-generation/sample_data \
  --pipeline-config=tutorials/nemo-retriever-synthetic-data-generation/config/config.yaml\
  --input-format=jsonl \
  --pipeline-type=generate \
  --output-dir=tutorials/nemo-retriever-synthetic-data-generation/outputs/sample_data_rawdoc
  --save-format=jsonl
  --n-partitions=5
```
The data can be saved in two formats (1. jsonl, 2. beir). Additionally, the user can pass ```--n-partitions``` flag to speed-up generation for large datasets.

To filter pre-generated data, run ```main.py``` with ```--pipeline-type=filter```
Note the change in the ```input-dir```, we need to use the path to the generated data in jsonl format.
```
python tutorials/nemo-retriever-synthetic-data-generation/main.py \
  --api-key=<API Key> \
  --input-dir= tutorials/nemo-retriever-synthetic-data-generation/outputs/sample_data_rawdoc/jsonl \
  --pipeline-config=tutorials/nemo-retriever-synthetic-data-generation/config/config.yaml\
  --input-format=jsonl \
  --pipeline-type=filter \
  --output-dir=tutorials/nemo-retriever-synthetic-data-generation/outputs/sample_data_rawdoc
  --save-format=jsonl
```

For more information about the expected structure of the data, see the [quick start notebook](notebooks/quickstart.ipynb).


### Using Custom Configuration

Edit [config.yaml](config/config.yaml) to update the configuration. Predefined configuration files can be found in [scripts/conf](config/config.yaml).


## Quality Improvement Playbook (for Advanced Users)


The default config file [config.yaml](config/config.yaml) should work best to generate synthetic data. You would need to change the few-shot examples in the prompt for specific use-cases. In case you'd like to improve the quality of synthetic data and/or apply the SDG pipeline for other domains, consider applying the recipes described below.


### Prompt templates

We recommend engineering the prompt templates for better synthetic data generations. Specifically, we have observed Chain-of-Thought prompting to result in the better generations as well. We have provided additional config files ([config-nq.yaml](config/config-nq.yaml) and [config-fiqa.yaml](config/config-fiqa.yaml)) that showcase Chain-of-Thought prompting.

Furthermore, they also showcase the use of in-context learning, wherein passage, query pairs were picked from datasets to be used as few-shot examples. Both methods yields good quality results.


### Choice of Easiness Filter & Threshold

We provide the embedding-model-as-a-judge as well as filter threshold value in our default configuration. The general recommendation to increase the difficulty of questions is to lower the filter threshold value and vice versa. The user can experiment with different filter threshold values to get more challenging or easier synthetic questions in their synthetic datasets.

The choice of the embedding model is provided in the default configuration. We experimented and verified the quality of the pipeline with the default configuration on multiple datasets such as FiQA, NQ and other internal datasets. The user can also change the embedding-model-as-a-judge by choosing any embedding model from [Huggingface Model Hub](https://huggingface.co/models).


### Choice of Answerability Filter

For Answerability Filter, our recommendation is to go with the choice provided in the default configuation file. We confirmed that the checkbox-style prompt in the default configuration worked well for valid question filtering.

However, the framework is flexible of the choice of LLM-as-a-Judge and different LLMs with different prompt templates might work better for certain use cases. You can also experiment with Likert-scale prompting if need be.

## Hard Negative Mining:
Hard-negative mining involves two steps. First step is to repartition the dataset into semantically similar documents. This is done using the following script,
```
python tutorials/nemo-retriever-synthetic-data-generation/repartition.py \
  --api-key=<API Key> \
  --input-dir=tutorials/nemo-retriever-synthetic-data-generation/sample_data/hard-neg-mining\
  --hard-negative-mining-config=tutorials/nemo-retriever-synthetic-data-generation/config/hard-negative-mining-config.yaml
  --output-dir=tutorials/nemo-retriever-synthetic-data-generation/my_clustered_dataset_dir
```
Once, the semantic clusters have been created, one can perform the hard negative mining as follows,
```
python tutorials/nemo-retriever-synthetic-data-generation/mine_hard_negatives.py \
  --api-key=<API Key> \
  --input-dir=tutorials/nemo-retriever-synthetic-data-generation/my_clustered_dataset_dir\
  --hard-negative-mining-config=tutorials/nemo-retriever-synthetic-data-generation/config/hard-negative-mining-config.yaml
  --output-dir=tutorials/nemo-retriever-synthetic-data-generation/my_mined_dataset_dir
```
