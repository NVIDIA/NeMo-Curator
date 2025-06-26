"""Example text processing pipeline using ray-curator."""

import argparse
import json
from pathlib import Path
from pprint import pprint
from typing import Literal

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter, ParquetWriter


def create_sample_jsonl_files(output_dir: Path, num_files: int = 3) -> None:
    """Create sample JSONL files for testing."""
    output_dir.mkdir(exist_ok=True)

    # Sample text data
    sample_texts = [
        "This is the first sample document with some interesting content about machine learning.",
        "Here's another document discussing natural language processing and text analysis.",
        "The third document covers topics related to data processing and pipeline optimization.",
        "Document four explores the world of distributed computing and parallel processing.",
        "Fifth document delves into the realm of artificial intelligence and deep learning.",
        "Sixth document examines big data technologies and their applications in modern industry.",
        "The seventh document discusses cloud computing and scalable architectures.",
        "Document eight covers database management and data storage solutions.",
        "Ninth document explores web development and modern frontend technologies.",
        "The final document discusses software engineering best practices and methodologies.",
    ]
    # Create multiple JSONL files
    docs_per_file = len(sample_texts) // num_files

    for file_idx in range(num_files):
        file_path = output_dir / f"sample_data_{file_idx}.jsonl"

        with open(file_path, "w") as f:
            start_idx = file_idx * docs_per_file
            end_idx = start_idx + docs_per_file if file_idx < num_files - 1 else len(sample_texts)

            for doc_idx in range(start_idx, end_idx):
                if doc_idx < len(sample_texts):
                    doc = {
                        "adlr_id": f"doc_{file_idx}_{doc_idx}",
                        "text": sample_texts[doc_idx],
                        "file_source": f"sample_data_{file_idx}.jsonl",
                        "doc_length": len(sample_texts[doc_idx]),
                    }
                    f.write(json.dumps(doc) + "\n")

        print(f"Created {file_path} with {end_idx - start_idx} documents")


def create_text_processing_pipeline(
    input_dir: Path, output_dir: Path, output_format: Literal["parquet", "jsonl"]
) -> Pipeline:
    """Create a pipeline for processing text data.
    The reader stage will create multiple tasks based on the partitioning
    strategy, which will then be processed in parallel.
    """

    # Define pipeline
    pipeline = Pipeline(name="text_processing", description="Process text data from JSONL files")

    # Add stages
    # The JsonlReader will create multiple tasks based on files_per_partition
    pipeline.add_stage(
        JsonlReader(
            file_paths=str(input_dir),  # Read from our created sample files
            files_per_partition=2,  # Each task will process 2 files
            reader="pandas",  # Use pandas reader
        )
    )
    if output_format == "jsonl":
        writer = JsonlWriter(output_dir=str(output_dir))
    elif output_format == "parquet":
        writer = ParquetWriter(output_dir=str(output_dir))
    else:
        msg = f"Invalid output format: {output_format}"
        raise ValueError(msg)

    pipeline.add_stage(writer)

    return pipeline


def main(args: argparse.Namespace) -> None:
    """Main function to run the pipeline."""

    # Create sample data directory
    input_dir = Path(args.input_path).resolve()

    # Create sample JSONL files
    print("Creating sample JSONL files...")
    create_sample_jsonl_files(input_dir, num_files=3)
    print(f"Sample files created in: {input_dir}")
    print("\n" + "=" * 50 + "\n")

    # Create pipeline
    pipeline = create_text_processing_pipeline(input_dir, Path(args.output_path).resolve(), args.output_format)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    if args.executor == "xenna":
        executor = XennaExecutor()
    elif args.executor == "ray_data":
        from ray_curator.backends.experimental.ray_data import RayDataExecutor

        executor = RayDataExecutor()
    elif args.executor == "ray_actors":
        from ray_curator.backends.experimental.ray_actors import RayActorExecutor

        executor = RayActorExecutor()
    else:
        msg = f"Invalid executor type: {args.executor}"
        raise ValueError(msg)

    # Execute pipeline
    print("Starting pipeline execution...")
    results = pipeline.run(executor)

    # Print results summary
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    total_documents = sum(task.num_items for task in results) if results else 0
    print(f"Total documents processed: {total_documents}")

    # Save results
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(exist_ok=True)

    if results:
        print("\nSample of processed documents:")
        for task in results:
            pprint(task)
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./sample_jsonl_data")
    parser.add_argument("--output_path", type=str, default="./test_output")
    parser.add_argument("--output_format", type=str, default="parquet", choices=["parquet", "jsonl"])

    parser.add_argument("--executor", type=str, default="xenna", choices=["xenna", "ray_data", "ray_actors"])
    args = parser.parse_args()

    main(args)
