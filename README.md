# NeMo Curator

NeMo Curator is a Python library designed for scalable and efficient dataset preparation, enhancing LLM training accuracy through GPU-accelerated data curation using [Dask](https://www.dask.org/) and [RAPIDS](https://developer.nvidia.com/rapids). It offers a customizable and modular interface that simplifies pipeline expansion and accelerates model convergence by preparing high-quality tokens.

At the core of the NeMo Curator is the DocumentDataset which serves as the the main dataset class. It acts as a straightforward wrapper around a Dask dataframe. The Python library offers easy-to-use methods for expanding the functionality of your curation pipeline while eliminating scalability concerns.

## Key Features

NeMo Curator provides a collection of scalable data-mining modules. Some of the key features include:

[Data download and text extraction](docs/user-guide/Download.rst)

- Default implementations for downloading and extracting Common Crawl, Wikipedia, and ArXiv data
- Easily customize the download and extraction and extend to other datasets

[Language identification and separation](docs/user-guide/LanguageIdentificationUnicodeFormatting.rst)

- Language identification with [fastText](https://fasttext.cc/docs/en/language-identification.html) and [pycld2](https://pypi.org/project/pycld2/)

[Text reformatting and cleaning](docs/user-guide/LanguageIdentificationUnicodeFormatting.rst)

- Fix unicode decoding errors via [ftfy](https://ftfy.readthedocs.io/en/latest/)

[Quality filtering](docs/user-guide/QualityFiltering.rst)

- Multilingual heuristic-based filtering
- Classifier-based filtering via [fastText](https://fasttext.cc/)

[Document-level deduplication](docs/user-guide/GpuDeduplication.rst)

- Both exact and fuzzy deduplication are accelerated using cuDF and Dask
- For fuzzy deduplication, our implementation follows the method described in [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990)

[Multilingual downstream-task decontamination](docs/user-guide/TaskDecontamination.rst)

- Our implementation follows the approach of [OpenAI GPT3](https://arxiv.org/pdf/2005.14165.pdf) and [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990)

[Distributed data classification](docs/user-guide/DistributedDataClassification.rst)

- Multi-node, multi-GPU classifier inference
- Provides sophisticated domain and quality classification
- Flexible interface for extending to your own classifier network

[Personal identifiable information (PII) redaction](docs/user-guide/PersonalIdentifiableInformationIdentificationAndRemoval.rst)

- Identification tools for removing addresses, credit card numbers, social security numbers, and more

These modules offer flexibility and permit reordering, with only a few exceptions. In addition, the [NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) provides pre-built pipelines that can serve as a foundation for your customization use cases.

## Resources

- [Documentation](docs/)
- [Examples](examples/)
- [Tutorials](tutorials/)

## Get Started

This section explains how to install NeMo Curator and use the Python library, Python modules, and CLI scripts. It also includes a list of tutorials to help you get started right away. Finally, this section explains how to use the NeMo Framework Launcher as an alternative method for interfacing with NeMo Curator.

### Install NeMo Curator

NeMo Curator is available in the [NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). The latest release of NeMo Curator comes preinstalled in the container.

Before installing NeMo Curator, ensure that the following requirements are met:

+- Python 3.10 (or above)
+- CUDA 12 (or above)
+- NVIDIA GPU (optional)

First, clone the NeMo Curator repository in GitHub.

Next, install the modules that you need.

To install the CPU-only modules:

```
pip install
```

To install the CPU and CUDA-accelerated modules:
```
pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
```

### Use the Python Library

To download your dataset, build your pipeline, and curate your dataset:

```Python
# Download your dataset
dataset = download_common_crawl("/datasets/common_crawl/", "2021-04", "2021-10", url_limit=10)
# Build your pipeline
curation_pipeline = Sequential([
  Modify(UnicodeReformatter()),
  ScoreFilter(WordCountFilter(min_words=80)),
  ScoreFilter(FastTextQualityFilter(model_path="model.bin")),
  TaskDecontamination([Winogrande(), Squad(), TriviaQA()])
])
# Curate your dataset
curated_dataset = curation_pipeline(dataset)
```

### Explore NeMo Curator Tutorials

To get started with NeMo Curator, you can follow the tutorials available here: [Tutorials]
(https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials). These tutorials include:

+- tinystories which focuses on data curation for training from scratch.
+- peft-curation which focuses on data curation for parameter-efficient fine-tuning use-cases.

### Python Modules

NeMo Curator provides a collection of robust Python modules that you can chain together to construct your entire data curation pipeline. You can run these modules on your local machine or in a distributed compute environment like SLURM without the need to make modifications.

NeMo Curator also offers simple base classes for inheritance, enabling you to develop your own filters, document modifiers, and additional extensions without the concern of scalability.

The [examples](examples/) directory contains scripts that showcase each of these modules. The Data Curation section of the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html) provides in-depth information on how each of the modules work. If you need more information about how to modify NeMo Curator for your use case, see [Implement NeMo Curator](#implement-nemo-curator).

### CLI Scripts

NeMo Curator also offers CLI scripts for you to use. The scripts in `nemo_curator/scripts` map closely to the supplied Python modules. Refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html) for more information about the Python modules and scripts.

### NeMo Framework Launcher

As an alternative method for interfacing with NeMo Curator, you can use the [NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher). The launcher enables you to easily configure the parameters and cluster. It can also automatically generate the SLURM batch scripts that wrap around the CLI scripts required to run your pipeline.

Note: Other methods are available to run NeMo Curator on SLURM. For example, refer to the example scripts in [`examples/slurm`](examples/slurm/) for information on how to run NeMo Curator on SLURM without the NeMo Framework Launcher.

## Module Ablation and Compute Performance

The modules within NeMo Curator were primarily designed to curate high-quality documents from Common Crawl snapshots in a scalable manner. To evaluate the quality of the curated Common Crawl documents, we conducted a series of ablation experiments. In these experiments, we trained a 357M-parameter GPT-style model using datasets generated at various stages of our data curation pipeline, which was implemented in NeMo Curator. 

The following figure shows that the use of different data curation modules implemented in NeMo Curator led to improved model zero-shot downstream task performance.

<p align="center">
  <img src="./docs/user-guide/images/zeroshot_ablations.png" alt="drawing" width="700"/>
</p>

In terms of scalability and compute performance, using the combination of RAPIDS and Dask fuzzy deduplication enabled us to deduplicate the 1.1 Trillion token Red Pajama dataset in 1.8 hours with 64 NVIDIA A100 Tensor Core GPUs.

Additionally, using the CPU-based modules, the following table shows the time required and resulting data size reduction for each processing step [Common Crawl snapshot from November/December of 2020](https://commoncrawl.org/2020/12/nov-dec-2020-crawl-archive-now-available/) using 30 CPU nodes (with hardware similar to the `c5.24xlarge` [Amazon AWS C5 instance](https://aws.amazon.com/ec2/instance-types/c5/)).

<table>
  <thead>
    <tr>
      <th style="text-align:center">Dataset </th>
      <th colspan="2"> Download and text extraction</th>
      <th colspan="2">Text cleaning </th>
      <th colspan="2">Quality filtering</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Time  </td>
      <td> Output Size </td>
      <td>Time </td>
      <td> Output Size </td>
      <td>Time </td>
      <td> Output Size </td>
    </tr>
    <tr>
      <td>Common Crawl 2020-50</td>
      <td> 36 hrs</td>
      <td>2.8 TB</td>
      <td> 1 hr </td>
      <td> 2.8 TB </td>
      <td> 0.2 hr </td>
      <td> 0.52 TB </td>
    </tr>
  </tbody>
</table>

## Contribute to NeMo

We welcome community contributions! Please refer to `CONTRIBUTING.md <https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md>`_ for the process.

To contribute an article to the collection, please submit a pull request to the ``gh-pages-src`` branch of this repository. For detailed information, please consult the README located at the `gh-pages-src branch <https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme>`_.