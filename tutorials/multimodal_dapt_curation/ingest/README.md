# Multi-Modal Data Extraction from PDFs

## Overview
This tutorial guides you through extracting different modalities (text, images, tables, etc.) from PDFs using NVIDIA's multimodal extraction (`nv-ingest`) framework.

## Setup
Before proceeding, ensure you have completed the setup instructions provided in the following GitHub repository:

[NVIDIA Ingest Multi-Modal Data Extraction](https://github.com/NVIDIA/nv-ingest?tab=readme-ov-file#nvidia-ingest-multi-modal-data-extraction)

This link includes essential steps required to run this tutorial.

## Installation
Once the setup is complete, follow these steps to install the required dependencies for running this tutorial:

```sh
pip install -r requirements.txt
```

## Running the Tutorial
After installing the dependencies, execute the following command to run the tutorial:

```sh
python main.py
```

## Optional Flags
The tutorial provides two optional flags:

- `--analyze`: Analyzes the extracted contents, identifying unique types and descriptions.
- `--display`: Displays extracted images or table contents as Base64 decoded files.

Example usage:

```sh
python main.py --analyze --display
```

This will generate a directory named `separated_extracted_data` under `sources` directory. This contains the extracted data mapped to modality from all the data sources. The `data_type_map.json` file will be used by NeMo-Curator to curate data for DAPT task.

## Notes
- Ensure your Python environment is activated before installing dependencies.
- The extracted content can be further processed using NeMo Curator for domain adapted pre-training.
- Each ingest job includes a set of tasks. These tasks define the operations that will be performed during ingestion. 
    - `extract` : Performs multimodal extractions from a document, including text, images, and tables.
    - `dedup` : Identifies duplicate images in document that can be filtered to remove data redundancy.
    - `filter` : Filters out images that are likely not useful using some heuristics, including size and aspect ratio.
    Ingestor interface helps to chain together an extraction tast and a deduplication task to ingest PDF. 
- Exploring Outputs
    - Text
        - `content` - The raw extracted content, text in this case - this section will always be populated with a successful job.
        - `content_metadata` - Describes the type of extraction and its position in the broader document - this section will always be populated with a successful job.
        - `source_metadata` - Describes the source document that is the basis of the ingest job.
        - `text_metadata` - Contain information about the text extraction, including detected language, among others - this section will only exist when `metadata['content_metadata']['type'] == 'text'`
    - Charts and Tables
        - `content` - The raw extracted content, a base64 encoded image of the extracted table in this case - this section will always be populated with a successful job.
        - `content_metadata` - Describes the type of extraction and its position in the broader document - this section will always be populated with a successful job.
        - `source_metadata` - Describes the source and storage path of an extracted table in an S3 compliant object store.
        - `table_metadata` - Contains the text representation of the table, positional data, and other useful elements - this section will only exist when `metadata['content_metadata']['type'] == 'structured'`.
        - Note, `table_metadata` will store chart and table extractions. The are distringuished by `metadata['content_metadata']['subtype']`
    - Images
        - `content` - The raw extracted content, a base64 encoded image extracted from the document in this case - this section will always be populated with a successful job.
        - `content_metadata` - Describes the type of extraction and its position in the broader document - this section will always be populated with a successful job.
        - `source_metadata` - Describes the source and storage path of an extracted image in an S3 compliant object store.
        - `image_metadata` - Contains the image type, positional data, and other useful elements - this section will only exist when `metadata['content_metadata']['type'] == 'image'`.

## License
Refer to the original repository for licensing details.

