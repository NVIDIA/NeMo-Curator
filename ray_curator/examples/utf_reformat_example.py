"""Example text processing pipeline using ray-curator."""

from pathlib import Path

import pyarrow.parquet as pq
from ray_curator.executors import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.readers.jsonl import JsonlReader
from ray_curator.stages.modify import ModifierStage
from ray_curator.modifiers.unicode_reformatter import UnicodeReformatter


def create_text_processing_pipeline() -> Pipeline:
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
            file_paths="/raid/prospector-lm/dedup_test_rapids/original/",  # Can be a directory
            text_column="text",  # Specify which column contains text
            id_column="adlr_id",  # Optional ID column
            additional_columns=[],  # Preserve these columns
            files_per_partition=5,  # Each task will process 5 files
            reader="pandas",  # Use pandas reader
        )
    )
    pipeline.add_stage(ModifierStage(UnicodeReformatter()))
    return pipeline


def main() -> None:
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
    executor = XennaExecutor(config={"batch_size": 1, "num_workers": 4})

    # Execute pipeline
    print("Starting pipeline execution...")
    print("\nNote: The reader stage will create multiple tasks based on the")
    print("partitioning strategy. These tasks will be processed in parallel")
    print("by the available workers.\n")

    results = executor.execute(execution_plan)

    # Print results summary
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results)}")

    total_documents = sum(task.num_items for task in results)
    print(f"Total documents processed: {total_documents}")

    # Save results
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)

    for i, batch in enumerate(results):
        output_file = output_dir / f"batch_{i}.parquet"
        # Convert to PyArrow and save
        table = batch.to_pyarrow()
        pq.write_table(table, output_file)
        print(f"Saved batch {i} with {batch.num_items} documents to {output_file}")

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