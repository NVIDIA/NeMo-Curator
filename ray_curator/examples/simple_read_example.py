"""Example text processing pipeline using ray-curator."""

import json
from pathlib import Path

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("Warning: pyarrow not available. Results will not be saved to parquet.")

from ray_curator.backends.xenna.executor import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.readers.jsonl import JsonlReader


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


def create_text_processing_pipeline(data_dir: Path) -> Pipeline:
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
            file_paths=str(data_dir),  # Read from our created sample files
            text_column="text",  # Specify which column contains text
            id_column=None,  # Optional ID column
            additional_columns=["file_source", "doc_length"],  # Preserve these columns
            files_per_partition=2,  # Each task will process 2 files
            reader="pandas",  # Use pandas reader
        )
    )
    return pipeline


def main() -> None:
    """Main function to run the pipeline."""

    # Create sample data directory
    data_dir = Path("/raid/praateekm/ayush-ray-curator/sample_jsonl_data")

    # Create sample JSONL files
    print("Creating sample JSONL files...")
    create_sample_jsonl_files(data_dir, num_files=3)
    print(f"Sample files created in: {data_dir}")
    print("\n" + "=" * 50 + "\n")

    # Create pipeline
    pipeline = create_text_processing_pipeline(data_dir)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    print("\nNote: The reader stage will create multiple tasks based on the")
    print("partitioning strategy. These tasks will be processed in parallel")
    print("by the available workers.\n")

    results = pipeline.execute(executor)

    # Print results summary
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    total_documents = sum(task.num_items for task in results) if results else 0
    print(f"Total documents processed: {total_documents}")

    # Save results
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)

    for i, batch in enumerate(results or []):
        print(f"Processing batch {i} with {batch.num_items} documents...")

        # Show sample of data
        if batch.num_items > 0:
            df = batch.to_pandas()
            print(f"Columns: {list(df.columns)}")
            print(f"Text column: {batch.text_column}")

        # Save to parquet if pyarrow is available
        if HAS_PYARROW:
            output_file = output_dir / f"batch_{i}.parquet"
            table = batch.to_pyarrow()
            pq.write_table(table, output_file)
            print(f"Saved batch {i} to {output_file}")
        else:
            # Save to CSV as fallback
            output_file = output_dir / f"batch_{i}.csv"
            df = batch.to_pandas()
            df.to_csv(output_file, index=False)
            print(f"Saved batch {i} to {output_file}")

        print()


if __name__ == "__main__":
    main()
