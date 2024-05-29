# Data curation for DAPT (Domain Adaptive Pre-Training) 

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain adapted LLM. LLama models are continually pre-trained with 20B plus tokens on domain-specific chip design data, including code, documents, etc., based on NeMo foundation models and then fine-tuned with instruction datasets from design data as well as external sources. 

Here, we share playbooks with best practices on DAPT (domain-adaptive pre-training) for a ChipNeMo-like code generation use case.
* `\llm_data_collection` contains playbooks and scripts for DAPT data curation
* `\pdfcleaner` contains library and scripts need to convert pdfs to txt files

* Playbook for data curation with NeMo Curator: `\llm_data_collection\notebooks\data_curation-nemo-curator_DAPT.ipynb`
* Playbook for data curation without NeMo Curator: `\llm_data_collection\notebooks\data_curation_DAPT.ipynb`
* Playbook for data curation of pdfs (with conversion to txt): `\llm_data_collection\notebooks\data_curation_pdf.ipynb`