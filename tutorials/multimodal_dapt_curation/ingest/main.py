import argparse
import json
import os
import re
import shutil
import time
from base64 import b64decode
from collections import defaultdict
from io import BytesIO

import arxiv as arxiv
import pandas as pd
from nv_ingest_client.client import Ingestor
from PIL import Image

# Constants for configuration and paths
HTTP_HOST = os.environ.get("HTTP_HOST", "localhost")
HTTP_PORT = os.environ.get("HTTP_PORT", "7670")
TASK_QUEUE = os.environ.get("TASK_QUEUE", "morpheus_task_queue")


DEFAULT_JOB_TIMEOUT = 10000  # Timeout for job completion (in ms)

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "sources")
RESULT_DIR = os.path.join(DATA_DIR, "extracted_data")
OUTPUT_DIR = os.path.join(DATA_DIR, "separated_extracted_data")


def parse_id(input_string):
    """
    Parse arXiv ID from either a direct ID string or an arXiv URL.
    """
    # Pattern to match a direct arXiv ID
    id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
    if id_pattern.match(input_string):
        return input_string

    # Pattern to match an arXiv URL and extract the ID
    url_pattern = re.compile(
        r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$"
    )
    url_match = url_pattern.match(input_string)
    if url_match:
        return url_match.group(2) + (url_match.group(3) if url_match.group(3) else "")

    # Raise an error if the input does not match any of the expected formats
    raise ValueError(
        f"The provided input '{input_string}' does not match the expected URL or ID format."
    )


def download_arxiv_data():
    """
    Download arXiv articles from URLs listed in arxiv_urls.jsonl and save them as PDFs.
    """
    pdf_root_dir = os.path.join(DATA_DIR, "pdfs")
    os.makedirs(pdf_root_dir, exist_ok=True)

    source_links_file = os.path.join(DATA_DIR, "arxiv_urls.jsonl")
    if not os.path.exists(source_links_file):
        raise FileNotFoundError(f"File '{source_links_file}' not found.")

    urls = pd.read_json(path_or_buf=source_links_file, lines=True)
    urls = urls[0].tolist()

    for url in urls:
        pdf_name = os.path.basename(url)
        pdf_file = os.path.join(pdf_root_dir, pdf_name)

        if os.path.exists(pdf_file):
            print(f"Article '{url}' already exists, skipping download.")
        else:
            article_id = parse_id(url)
            search_result = arxiv.Client().results(arxiv.Search(id_list=[article_id]))

            if article := next(search_result):
                print(f'Downloading arXiv article "{url}"...')
                article.download_pdf(dirpath=pdf_root_dir, filename=pdf_name)
            else:
                print(f"Failed to download article '{url}'.")
                return None


def separate_extracted_contents():
    jsonl_text_list = []
    jsonl_image_list = []
    jsonl_structured_list = []

    data_type_map = {
        "text": jsonl_text_list,
        "image": jsonl_image_list,
        "structured": jsonl_structured_list,
    }

    for file in os.listdir(RESULT_DIR):
        file_path = os.path.join(RESULT_DIR, file)
        with open(file_path, "r") as f:
            jsonl_loaded = json.load(f)
            for jsonl_chunk in jsonl_loaded:
                data_type = jsonl_chunk.get("document_type")
                if data_type in data_type_map:
                    data_type_map[data_type].append(jsonl_chunk)
                else:
                    print(
                        f"Unknown document type {data_type}. Add code to support this type."
                    )

    print(
        f"Processed {len(data_type_map['text'])} text chunks, {len(data_type_map['image'])} image chunks and {len(data_type_map['structured'])} table/chart chunks"
    )

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/data_type_map.json", "w", encoding="utf-8") as f:
        json.dump(data_type_map, f, indent=4, ensure_ascii=False)


def extract_contents():
    """
    Extract contents from downloaded PDFs and send them for processing.
    """
    downloaded_path = os.path.join(DATA_DIR, "pdfs")

    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR, exist_ok=True)

    for file in os.listdir(downloaded_path):
        print(f"Extracting contents from file {file}")
        sample_pdf = os.path.join(downloaded_path, file)

        ingestor = (
            Ingestor(message_client_hostname=HTTP_HOST)
            .files(sample_pdf)
            .extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=True,
                text_depth="document",
            )
            .dedup(
                content_type="image",
                filter=True,
            )
            .caption()
        )

        generated_metadata = ingestor.ingest()[0]

        # Save extracted metadata
        with open(
            os.path.join(RESULT_DIR, f"generated_metadata_{file}.json"), "w"
        ) as f:
            json.dump(generated_metadata, f, indent=4)


def analyze_contents():
    """
    Analyze the extracted contents for unique types and descriptions.
    """
    unique_types = set()
    unique_desc = set()
    type_indices = defaultdict(list)

    # Iterate through the generated metadata files
    for file in os.listdir(RESULT_DIR):
        file_path = os.path.join(RESULT_DIR, file)
        with open(file_path, "r") as f:
            generated_metadata = json.load(f)

        for i in range(len(generated_metadata)):
            description = generated_metadata[i]["metadata"]["content_metadata"][
                "description"
            ]
            unique_desc.add(description)

            content_type = generated_metadata[i]["metadata"]["content_metadata"]["type"]
            unique_types.add(content_type)
            type_indices[content_type].append(i)

    print("Unique types:", *unique_types, sep="\n")
    print("Number of contents extracted")
    for content_type, indices in type_indices.items():
        print(f"{content_type}: {len(indices)}")


def display_contents():
    """
    Display the extracted image or table contents as base64 decoded files.
    """

    output_folder = os.path.join(DATA_DIR, "extracted_display_data")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(RESULT_DIR):
        file_path = os.path.join(RESULT_DIR, file)
        with open(file_path, "r") as f:
            generated_metadata = json.load(f)

        for idx, metadata in enumerate(generated_metadata):
            content = metadata["metadata"]["content_metadata"]["type"]
            if content in ["image", "table"]:
                # Decode and save image data
                image_data_b64 = metadata["metadata"]["content"]
                image_data = b64decode(image_data_b64)
                image = Image.open(BytesIO(image_data))
                image_filename = os.path.join(output_folder, f"{file}_{idx}.png")
                image.save(image_filename, format="PNG")


def main():
    """
    Main function to execute the workflow with optional analysis and display.
    """
    parser = argparse.ArgumentParser(
        description="Execute workflow with optional analysis and display."
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Enable analysis of contents"
    )
    parser.add_argument(
        "--display", action="store_true", help="Enable displaying of contents"
    )

    args = parser.parse_args()

    download_arxiv_data()
    extract_contents()
    separate_extracted_contents()

    if args.analyze:
        analyze_contents()

    if args.display:
        display_contents()


if __name__ == "__main__":
    main()
