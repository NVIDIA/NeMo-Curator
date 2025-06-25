"""Ray Data adapter for processing stages."""

from typing import Any

from loguru import logger
from ray.data import Dataset

from ray_curator.backends.base import BaseStageAdapter
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import Task

from .setup_utils import setup_stage_with_coordination


class RayDataStageAdapter(BaseStageAdapter):
    """Adapts ProcessingStage to Ray Data operations.

    This adapter converts stages to work with Ray Data datasets by:
    1. Converting Task objects to/from dictionaries
    2. Using Ray Data's map_batches for parallel processing
    3. Handling single and batch processing modes
    4. Supporting setup() and setup_on_node() calls like other backends
    """

    def __init__(self, stage: ProcessingStage):
        super().__init__(stage)

        self._batch_size = self.stage.batch_size
        if self._batch_size is None and self.stage.resources.gpus > 0:
            logger.warning(f"When using Ray Data, batch size is not set for GPU stage {self.stage}. Setting it to 1.")
            self._batch_size = 1

    @property
    def batch_size(self) -> int | None:
        """Get the batch size for this stage."""
        return self._batch_size

    def _setup_if_needed(self) -> None:
        """Setup the stage if it hasn't been setup yet.

        This method ensures setup happens exactly once per Ray Data worker,
        and setup_on_node happens exactly once per node.
        """
        if not hasattr(self, "_setup_done") or not getattr(self, "_setup_done", False):
            # Mark setup as done first to avoid recursion
            self._setup_done = True

            # Use the setup utilities for coordinated setup
            setup_stage_with_coordination(stage=self.stage, setup_fn=self.setup, setup_on_node_fn=self.setup_on_node)

    def _process_batch_internal(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Internal method that handles the actual batch processing logic.

        Args:
            batch: Dictionary with arrays/lists representing a batch of tasks

        Returns:
            Dictionary with arrays/lists representing processed tasks
        """
        # Ensure setup is called before processing
        self._setup_if_needed()

        # Convert batch format from Ray Data to list of task dictionaries
        batch_size = len(next(iter(batch.values())))
        task_dicts = []

        for i in range(batch_size):
            task_dict = {key: values[i] for key, values in batch.items()}
            task_dicts.append(task_dict)

        # Convert dictionaries to Task objects
        tasks = [Task.from_dict(task_dict) for task_dict in task_dicts]

        results = self.process_batch(tasks)

        # Convert Task objects back to dictionaries
        result_dicts = [task.to_dict() for task in results]

        # # Convert list of dictionaries back to batch format
        batch_keys = next(iter(result_dicts)).keys()

        result_batch = {}
        for key in batch_keys:
            result_batch[key] = [result_dict.get(key) for result_dict in result_dicts]

        return result_batch

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a Ray Data dataset through this stage.

        Args:
            dataset (Dataset): Ray Data dataset containing task dictionaries

        Returns:
            Dataset: Processed Ray Data dataset
        """
        if self._is_fanout_stage():
            # For fanout stages, use flat_map to ensure proper parallelization
            # flat_map naturally distributes the fanout results across multiple blocks
            processed_dataset = dataset.flat_map(
                create_named_ray_data_stage_adapter(self.stage).flat_map_fn,
                num_cpus=self.stage.resources.cpus,
                num_gpus=self.stage.resources.gpus,
            )

            # After fanout, repartition with target_num_rows_per_block=1
            # to ensure each output item becomes a separate block for proper parallelization
            processed_dataset = processed_dataset.repartition(target_num_rows_per_block=1)

            # Force materialization to ensure the repartitioning actually happens
            # This creates separate blocks that can be processed in parallel by subsequent stages
            processed_dataset = processed_dataset.materialize()
        else:
            # Use Ray Data's map_batches for parallel processing
            processed_dataset = dataset.map_batches(
                create_named_ray_data_stage_adapter(self.stage).map_batch_fn,
                batch_size=self.batch_size,
                num_cpus=self.stage.resources.cpus,
                num_gpus=self.stage.resources.gpus,
            )

        return processed_dataset

    def map_batch_fn(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Map function that processes a batch of task dictionaries.
        This gets overriden by create_named_ray_data_stage_adapter
        """
        return self._process_batch_internal(batch)

    def _is_fanout_stage(self) -> bool:
        """Check if this stage is a fanout stage (returns list from process method).
        A fanout stage is one where process() can return a list[Task] instead of a single Task,
        meaning one input task can produce multiple output tasks.

        This method uses type annotation inspection of the process method return type
        """
        from typing import Union, get_args, get_origin

        # For debugging, log the stage being checked
        logger.debug(f"Checking if stage {self.stage.__class__.__name__} is a fanout stage")

        # Method 1: Inspect type annotations of the process method
        try:
            process_method = self.stage.process
            if hasattr(process_method, "__annotations__"):
                return_annotation = process_method.__annotations__.get("return")
                if return_annotation is not None:
                    logger.debug(f"Stage {self.stage.__class__.__name__} has return annotation: {return_annotation}")

                    # Check if return type is Union[Y, list[Y]] or list[Y]
                    origin = get_origin(return_annotation)
                    if origin is Union:
                        # Check if any of the union types is a list
                        args = get_args(return_annotation)
                        for arg in args:
                            if get_origin(arg) is list:
                                logger.debug(
                                    f"Stage {self.stage.__class__.__name__} detected as fanout via Union type annotation"
                                )
                                return True
                    elif origin is list:
                        # Direct list return type
                        logger.debug(
                            f"Stage {self.stage.__class__.__name__} detected as fanout via direct list return type"
                        )
                        return True
        except Exception as e:  # noqa: BLE001
            # If type inspection fails, continue to other methods
            logger.debug(f"Type annotation inspection failed for {self.stage.__class__.__name__}: {e}")

        logger.debug(f"Stage {self.stage.__class__.__name__} determined to NOT be a fanout stage")
        return False

    def flat_map_fn(self, task_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Flat map function for fanout stages that processes a single task dictionary and returns a list.
        This gets overridden by create_named_ray_data_stage_adapter for fanout stages
        """
        # Ensure setup is called before processing
        self._setup_if_needed()

        # Convert dictionary to Task object
        task = Task.from_dict(task_dict)

        # For fanout stages, call process() directly to get the list result
        # Don't use process_batch() as it flattens the results
        result = self.stage.process(task)

        # Ensure result is a list (fanout stages should return lists)
        if not isinstance(result, list):
            result = [result]

        # Convert Task objects back to dictionaries
        return [task.to_dict() for task in result]


def create_named_ray_data_stage_adapter(stage: ProcessingStage) -> RayDataStageAdapter:
    """Create a named Ray Data stage adapter.

    This creates an adapter instance and assigns a dynamically named map function
    that reflects the stage name, similar to the _create_named_map_function logic.

    Args:
        stage (ProcessingStage): Processing stage to adapt

    Returns:
        RayDataStageAdapter: Ray Data stage adapter with dynamically named map function
    """
    # Create the adapter instance
    adapter = RayDataStageAdapter(stage)

    # Get the stage name for the function
    stage_name = stage.__class__.__name__

    # Create a dynamically named map function
    def stage_map_fn(batch: dict[str, Any]) -> dict[str, Any]:
        """Dynamically named map function that processes a batch of task dictionaries."""
        return adapter._process_batch_internal(batch)

    # Set the function name to include the stage name
    stage_map_fn.__name__ = f"{stage_name}"
    stage_map_fn.__qualname__ = f"{stage_name}"

    # Assign the dynamically named function to the adapter
    adapter.map_batch_fn = stage_map_fn

    # Save reference to original flat_map_fn before overwriting it
    original_flat_map_fn = adapter.flat_map_fn

    # Create a dynamically named flat_map function for fanout stages
    def stage_flat_map_fn(task_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Dynamically named flat map function for fanout stages."""
        # Call the original flat_map_fn to avoid recursion
        return original_flat_map_fn(task_dict)

    # Set the function name to include the stage name
    stage_flat_map_fn.__name__ = f"{stage_name}_flat_map"
    stage_flat_map_fn.__qualname__ = f"{stage_name}_flat_map"

    # Assign the dynamically named flat_map function to the adapter
    adapter.flat_map_fn = stage_flat_map_fn

    return adapter
