"""Example of using the Xenna executor with ray-curator pipeline framework."""

import logging

import pandas as pd

from ray_curator.data import DocumentBatch
from ray_curator.executors.xenna_executor import XennaExecutor
from ray_curator.pipeline.planner import ExecutionPlan
from ray_curator.readers.base import FileGroupTask
from ray_curator.stages.base import ProcessingStage, StageType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example custom stages using ray-curator's framework


class JsonlReaderStage(ProcessingStage[DocumentBatch]):
    """Read JSONL files and create document batches."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "jsonl_reader"

    @property
    def stage_type(self) -> StageType:
        return StageType.READER

    @property
    def cpu_cores(self) -> float:
        return 0.5  # IO-bound operation

    def process(self, task: FileGroupTask) -> list[DocumentBatch]:
        """Read JSONL files from FileGroupTask and create batches."""
        batches = []

        # Process each file in the file group
        for i, file_path in enumerate(task.file_paths):
            # In real implementation, read actual JSONL file
            data = {
                "id": [f"doc_{task.task_id}_{i}_{j}" for j in range(self.batch_size)],
                "content": [f"Sample text from {file_path}" for _ in range(self.batch_size)],
                "source": [file_path] * self.batch_size,
            }

            df = pd.DataFrame(data)

            batch = DocumentBatch(
                task_id=f"{task.task_id}_batch_{i}",
                dataset_name=task.dataset_name,
                data=df,
                text_column="content",
                id_column="id",
                additional_columns=["source"],
            )

            batches.append(batch)

        return batches


class HtmlExtractorStage(ProcessingStage[DocumentBatch]):
    """Extract text from HTML content."""

    @property
    def name(self) -> str:
        return "html_extractor"

    @property
    def stage_type(self) -> StageType:
        return StageType.EXTRACTOR

    @property
    def can_fuse_with(self) -> list[StageType]:
        return [StageType.FILTER]  # Can be fused with filters

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        """Extract text from HTML in documents."""
        df = task.to_pandas()

        # Simulate HTML extraction
        def extract_text(html_content: str) -> str:
            # In real implementation, use BeautifulSoup or similar
            return html_content.replace("<html>", "").replace("</html>", "")

        df[task.text_column] = df[task.text_column].apply(extract_text)

        # Update task with processed data
        task.data = df
        task.metadata["html_extracted"] = True

        return task


class DocumentFilterStage(ProcessingStage[DocumentBatch]):
    """Filter documents based on criteria."""

    def __init__(self, min_length: int = 100, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length

    @property
    def name(self) -> str:
        return "document_filter"

    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER

    @property
    def cpu_cores(self) -> float:
        return 0.5  # Light CPU work

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        """Filter documents by length."""
        df = task.to_pandas()

        # Apply length filter
        text_lengths = df[task.text_column].str.len()
        mask = (text_lengths >= self.min_length) & (text_lengths <= self.max_length)

        filtered_df = df[mask].copy()

        if len(filtered_df) == 0:
            return None  # No documents passed the filter

        # Update task
        task.data = filtered_df
        task.metadata["filtered"] = True
        task.metadata["original_count"] = len(df)
        task.metadata["filtered_count"] = len(filtered_df)

        return task


class TextEmbeddingStage(ProcessingStage[DocumentBatch]):
    """Generate embeddings using a transformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    @property
    def name(self) -> str:
        return "text_embedding"

    @property
    def stage_type(self) -> StageType:
        return StageType.TRANSFORMER

    @property
    def requires_gpu(self) -> bool:
        return True

    @property
    def gpu_memory_gb(self) -> float:
        return 8.0  # Typical for small embedding models

    def setup(self) -> None:
        """Load the embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        # In real implementation, load actual model
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(self.model_name)
        self.model = f"MockModel({self.model_name})"

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        """Generate embeddings for documents."""
        df = task.to_pandas()

        # Get text content
        texts = df[task.text_column].tolist()

        # Generate embeddings (mock implementation)
        embeddings = [[0.1] * 384 for _ in texts]  # Mock 384-dim embeddings

        # Add embeddings to dataframe
        df["embedding"] = embeddings

        # Update task
        task.data = df
        task.metadata["embeddings_generated"] = True
        task.additional_columns.append("embedding")

        return task


def create_example_pipeline():
    """Create an example text processing pipeline."""

    # Define stages
    stages = [
        JsonlReaderStage(batch_size=1000),
        HtmlExtractorStage(),
        DocumentFilterStage(min_length=100, max_length=5000),
        TextEmbeddingStage(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    ]

    # Create initial file group tasks
    initial_tasks = [
        FileGroupTask(
            task_id="file_group_1",
            dataset_name="example_dataset",
            data=["data/file1.jsonl", "data/file2.jsonl"],
            metadata={"group": 1},
        ),
        FileGroupTask(
            task_id="file_group_2",
            dataset_name="example_dataset",
            data=["data/file3.jsonl", "data/file4.jsonl"],
            metadata={"group": 2},
        ),
    ]

    pipeline = Pipeline(
        name="text_processing_example",
        stages=stages,
        initial_tasks=initial_tasks,
        graph={},
        fusion_info={},
    )

    # Create execution plan
    # execution_plan = ExecutionPlan(
    #     pipeline_name="text_processing_example", stages=stages, initial_tasks=initial_tasks, graph={}, fusion_info={}, decomposition_info={}
    # )

    return execution_plan


def main():
    """Run the example pipeline with Xenna executor."""

    # Create pipeline
    execution_plan = create_example_pipeline()

    # Configure Xenna executor
    executor_config = {
        "batch_size": 100,
        "logging_interval": 30,
        "slots_per_actor": 2,
        "ignore_failures": False,
        "execution_mode": "streaming",
        "autoscale_interval_s": 120,
        "cpu_allocation_percentage": 0.95,
    }

    # Create executor
    executor = XennaExecutor(config=executor_config)

    logger.info("Starting pipeline execution with Xenna...")
    logger.info(f"Pipeline: {execution_plan.pipeline_name}")
    logger.info(f"Stages: {[stage.name for stage in execution_plan.stages]}")
    logger.info(f"Initial file groups: {len(execution_plan.initial_tasks)}")

    # Show execution plan details
    logger.info("\n" + execution_plan.describe())

    try:
        # Execute pipeline
        results = executor.execute(execution_plan)

        logger.info("\nPipeline completed successfully!")
        logger.info(f"Output tasks: {len(results)}")

        # Process results
        for i, task in enumerate(results[:3]):  # Show first 3 results
            logger.info(f"\nTask {i}: {task.task_id}")
            logger.info(f"  Dataset: {task.dataset_name}")
            logger.info(f"  Num items: {task.num_items}")
            logger.info(f"  Stage history: {task.stage_history}")
            logger.info(f"  Metadata: {task.metadata}")

            if isinstance(task, DocumentBatch):
                df = task.to_pandas()
                logger.info(f"  Columns: {list(df.columns)}")
                logger.info(f"  Sample text: {df[task.text_column].iloc[0][:100]}...")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
