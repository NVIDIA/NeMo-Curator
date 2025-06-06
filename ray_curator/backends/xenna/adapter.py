"""Xenna executor for running pipelines using Cosmos-Xenna backend."""

from loguru import logger
from typing import Any, Optional, Union

import ray
from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.ray_utils.resources import Resources as XennaResources
from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.resources import Resources
from cosmos_xenna.ray_utils.resources import NodeInfo as XennaNodeInfo
from cosmos_xenna.ray_utils.resources import WorkerMetadata as XennaWorkerMetadata

from ray_curator.tasks import Task
from ray_curator.pipeline import Pipeline
from ray_curator.stages.base import ProcessingStage
from ray_curator.utils.performance_utils import StageTimer


class XennaStageAdapter(pipelines_v1.Stage):
    """Adapts ProcessingStage to Xenna.
    Args:
        stage: ProcessingStage to adapt
    """

    def __init__(self, processing_stage: ProcessingStage,):
        self.processing_stage = processing_stage
        self._timer = StageTimer(processing_stage)

    @property
    def required_resources(self) -> XennaResources:
        """Get the resources required for this stage."""
        logger.info(f"Resources: {self.processing_stage.resources}")
        return XennaResources(
            cpus=self.processing_stage.resources.cpus,
            gpus=self.processing_stage.resources.gpus,
            nvdecs=self.processing_stage.resources.nvdecs,
            nvencs=self.processing_stage.resources.nvencs,
            entire_gpu=self.processing_stage.resources.entire_gpu,
        )

    @property
    def stage_batch_size(self) -> int:
        """Get the batch size for this stage."""
        return self.processing_stage.batch_size

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv | None:
        """Runtime environment for this stage."""
        # Can be customized per stage if needed
        return None

    def process_data(self, tasks: list[Task]) -> Optional[list[Task]]:
        """Process batch of tasks with automatic performance tracking.
        Args:
            tasks: List of tasks to process
        Returns:
            List of processed tasks or None
        """
        if not tasks:
            return None

        # import pdb; pdb.set_trace()
        # Calculate input data size for timer
        input_size_bytes = sum(task.num_items for task in tasks)

        # Initialize performance timer for this batch
        self._timer.reinit(self.processing_stage, input_size_bytes)

        results = []

        # Wrap processing with performance timer
        with self._timer.time_process(len(tasks)):
            # Check if stage supports batch processing
            if self.processing_stage.supports_batch_processing():
                batch_results = self.processing_stage.process_batch(tasks)

                # Process the batch results
                for i, result in enumerate(batch_results):
                    if result is not None:
                        # Handle case where stage returns multiple tasks
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
            else:
                for task in tasks:
                    # Process task
                    result = self.processing_stage.process(task)

                    if result is not None:
                        # Handle case where stage returns multiple tasks
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)

        # Log performance stats and add to result tasks
        stage_name, stage_perf_stats = self._timer.log_stats()
        for task in results:
            task.add_stage_perf(stage_perf_stats)

        return results if results else None

    def setup_on_node(self, node_info: XennaNodeInfo, worker_metadata: XennaWorkerMetadata) -> None:
        """Setup the stage on a node - Xenna-specific signature.
        This method is called by Xenna with its specific types. We convert them
        to our generic types and delegate to the base adapter.
        Args:
            node_info: Xenna's NodeInfo object
            worker_metadata: Xenna's WorkerMetadata object
        """
        # Convert Xenna's types to our generic types (simplified)
        generic_node_info = NodeInfo(node_id=node_info.node_id)
        generic_worker_metadata = WorkerMetadata(
            worker_id=worker_metadata.worker_id,
            allocation=worker_metadata.allocation,  # Keep the original allocation object
        )
        self.processing_stage.setup_on_node(generic_node_info, generic_worker_metadata)

    def setup(self, worker_metadata: XennaWorkerMetadata) -> None:
        """Setup the stage per worker - Xenna-specific signature.
        This method is called by Xenna with its specific types. We convert them
        to our generic types and delegate to the base adapter.
        Args:
            worker_metadata: Xenna's WorkerMetadata object
        """
        # Convert Xenna's WorkerMetadata to our generic type
        generic_worker_metadata = WorkerMetadata(
            worker_id=worker_metadata.worker_id,
            allocation=worker_metadata.allocation,  # Keep the original allocation object
        )

        self.processing_stage.setup(generic_worker_metadata)


def create_named_xenna_stage_adapter(stage: ProcessingStage) -> XennaStageAdapter:
    """Create a XennaStageAdapter subclass with the same name as the wrapped stage.
    This ensures that when Xenna calls type(adapter).__name__, it returns the
    original stage's class name rather than 'XennaStageAdapter'.
    Args:
        stage: ProcessingStage to adapt
        batch_size: Number of tasks to process at once
    Returns:
        XennaStageAdapter instance with the wrapped stage's class name
    """
    # Get the original stage's class name
    original_class_name = type(stage).__name__

    # Create a dynamic subclass with the original name
    DynamicAdapter = type(  # noqa: N806
        original_class_name,  # Use the original stage's name
        (XennaStageAdapter,),  # Inherit from XennaStageAdapter
        {
            "__module__": XennaStageAdapter.__module__,  # Keep the same module
        },
    )

    # Create and return an instance of the dynamic adapter
    return DynamicAdapter(stage)