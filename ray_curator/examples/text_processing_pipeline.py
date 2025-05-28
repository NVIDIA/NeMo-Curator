"""Example text processing pipeline using ray-curator."""

import logging
from pathlib import Path

from ray_curator.executors import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages import (
    DocumentFilterStage,
    HtmlExtractorStage,
    JsonlReaderStage,
    ParquetReaderStage,
)
from ray_curator.stages.filters import content_contains_filter, language_filter, length_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def create_text_processing_pipeline():
    """Create a pipeline for processing text data.

    The reader stage will create multiple tasks based on the partitioning
    strategy, which will then be processed in parallel.
    """

    # Define pipeline
    pipeline = Pipeline(name="text_processing", description="Process text data from JSONL and Parquet files")

    # Add stages
    # The JsonlReaderStage will create multiple tasks based on files_per_partition
    pipeline.add_stage(
        JsonlReaderStage(
            file_paths="data/jsonl/",  # Can be a directory
            text_column="text",  # Specify which column contains text
            id_column="doc_id",  # Optional ID column
            additional_columns=["language", "source"],  # Preserve these columns
            files_per_partition=5,  # Each task will process 5 files
            reader="pandas",  # Use pandas reader
            storage_options={"anon": True},  # For S3 access
        )
    ).add_stage(HtmlExtractorStage(preserve_structure=True, remove_scripts=True, remove_styles=True)).add_stage(
        DocumentFilterStage(
            filters=[
                length_filter(min_length=100, max_length=10000, text_column="text"),
                language_filter(["en", "es", "fr"]),
                content_contains_filter(["data", "processing"], text_column="text"),
            ],
            filter_mode="all",
        )
    )

    return pipeline


def create_parquet_pipeline():
    """Create a pipeline for processing Parquet files.

    This example shows using blocksize-based partitioning.
    """

    pipeline = Pipeline(name="parquet_processing", description="Process text data from Parquet files")

    # The ParquetReaderStage will create tasks of approximately 128MB each
    pipeline.add_stage(
        ParquetReaderStage(
            file_paths="s3://my-bucket/data/*.parquet",  # Cloud paths supported
            text_column="content",
            id_column="id",
            blocksize="128MB",  # Create tasks of ~128MB each
            reader_kwargs={"columns": ["id", "content", "metadata"]},
            storage_options={"anon": True},
        )
    ).add_stage(DocumentFilterStage(filters=[length_filter(min_length=50, text_column="content")]))

    return pipeline


def demonstrate_reader_partitioning():
    """Demonstrate how readers create multiple tasks for parallel processing."""

    print("\n" + "=" * 50)
    print("Reader Partitioning Examples")
    print("=" * 50 + "\n")

    # Example 1: Files per partition
    reader1 = JsonlReaderStage(file_paths=[f"data/file_{i}.jsonl" for i in range(20)], files_per_partition=5)

    task_inputs = reader1.create_task_inputs()
    print("Files per partition = 5:")
    print(f"Created {len(task_inputs)} tasks from 20 files")
    for task_input in task_inputs:
        print(f"  Task {task_input['partition_idx']}: {task_input['num_files']} files")

    # Example 2: Blocksize partitioning
    reader2 = ParquetReaderStage(file_paths="data/large_dataset/", blocksize="256MB")

    print("\nBlocksize = 256MB:")
    print("Tasks will be created to process approximately 256MB each")
    print("The actual number of tasks depends on the total data size")


def main():
    """Main function to run the pipeline."""

    # Create pipeline
    pipeline = create_text_processing_pipeline()

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Build execution plan
    execution_plan = pipeline.build()

    # Print execution plan
    print(execution_plan.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor(
        config={
            "batch_size": 100,
            "num_workers": 4,
            "use_ray": True,  # Use Ray backend for now
        }
    )

    # For reader stages, we don't need initial data
    # The reader will generate tasks from files
    initial_tasks = []

    # Execute pipeline
    print("Starting pipeline execution...")
    print("\nNote: The reader stage will create multiple tasks based on the")
    print("partitioning strategy. These tasks will be processed in parallel")
    print("by the available workers.\n")

    results = executor.execute(execution_plan, initial_tasks)

    # Print results summary
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results)}")

    total_documents = sum(task.num_items for task in results)
    print(f"Total documents processed: {total_documents}")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for i, batch in enumerate(results):
        output_file = output_dir / f"batch_{i}.parquet"
        # Convert to PyArrow and save
        table = batch.to_pyarrow()
        # table.to_parquet(output_file)  # Uncomment to actually save
        print(f"Would save batch {i} with {batch.num_items} documents to {output_file}")

        # Show sample of data
        if batch.num_items > 0:
            df = batch.to_pandas()
            print(f"\nSample from batch {i}:")
            print(f"Columns: {list(df.columns)}")
            print(f"Text column: {batch.text_column}")

            # Get first document text
            text_series = batch.get_text_series()
            if len(text_series) > 0:
                print(f"First document preview: {text_series.iloc[0][:200]}...")


if __name__ == "__main__":
    main()
    # demonstrate_reader_partitioning()  # Uncomment to see partitioning examples
