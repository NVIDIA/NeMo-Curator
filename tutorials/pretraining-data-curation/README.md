# RedPajama-Data-v2 Datasets Curation for LLM Pretraining

This tutorial demonstrates the usage of NeMo Curator to curate the RedPajama-Data-v2 dataset for LLM pretraining.

## RedPajama-Data-v2
RedPajama-V2 is an open dataset for training large language models. The dataset includes over 100B text documents coming from 84 CommonCrawl snapshots and processed using the CCNet pipeline. In this tutorial, we will be perform data curation on two snapshots for demonstration purpuses.

## Getting Started
This tutorial is designed for multi-node environment and uses slurm for scheduling allocating resources. To start the tutorial, run the `start-distributed-notebook.sh` script in this directory which will start the Jupyter notebook that demonstrates the step by step walkthrough of the end to end curation pipeline. The notebook will run on port 8000 of the scheduler node. To work with the notenook locally, you can set up a SSH connection to the scheduler node.
