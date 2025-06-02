"""Pipeline runner for executing pipeline specifications."""

import logging
from typing import Any

from ray_curator.data import Task
from ray_curator.executors.xenna_executor import XennaExecutor
from ray_curator.pipeline.planner import ExecutionPlan, PipelinePlanner
from ray_curator.pipeline.spec import PipelineSpec
from ray_curator.readers.base import Reader
from ray_curator.stages.base import ProcessingStage

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs pipeline specifications by converting them to execution plans."""

    def __init__(self, executor_config: dict[str, Any] | None = None):
        """Initialize the pipeline runner.

        Args:
            executor_config: Configuration for the executor
        """
        self.executor_config = executor_config or {}
        self.planner = PipelinePlanner()

    def run(self, spec: PipelineSpec) -> list[Task]:
        """Run a pipeline specification.

        This method:
        1. Validates the pipeline spec
        2. Converts it to an execution plan
        3. Executes the plan and returns results

        Args:
            spec: Pipeline specification to run

        Returns:
            List of output tasks from the pipeline
        """
        logger.info(f"Running pipeline: {spec.name}")

        # Validate the spec
        spec.validate()

        # Convert spec to execution plan
        plan = self._create_execution_plan(spec)

        # Log the plan
        logger.info(f"\n{plan.describe()}")

        # Create executor
        executor = XennaExecutor(self.executor_config)

        # Execute the plan
        results = executor.execute(plan)

        logger.info(f"Pipeline completed with {len(results)} output tasks")
        return results

    def _create_execution_plan(self, spec: PipelineSpec) -> ExecutionPlan:
        """Convert a pipeline spec to an execution plan.

        This handles the transformation of readers into file groups
        and reader stages, while preserving the order of operations.

        Args:
            spec: Pipeline specification

        Returns:
            Execution plan ready for execution
        """
        # Collect all file groups from readers
        all_file_groups = []
        reader_to_stage: dict[Reader, ProcessingStage] = {}

        # Process each reader to create file groups and stages
        for stage in spec.stages:
            if isinstance(stage, Reader):
                # Create file groups
                file_groups = stage.create_file_groups()
                all_file_groups.extend(file_groups)

                # Get the corresponding reader stage
                stage_name = stage.get_reader_stage_name()
                stage_class = self.planner._reader_stage_map.get(stage_name)

                if stage_class:
                    reader_stage = stage_class()
                    reader_to_stage[stage] = reader_stage
                    logger.info(f"Created {len(file_groups)} file groups for {stage.__class__.__name__}")
                else:
                    raise ValueError(f"Unknown reader stage: {stage_name}")

        # Build the execution stages in the correct order
        execution_stages = []

        for stage in spec.stages:
            if isinstance(stage, Reader):
                # Replace reader with its processing stage
                execution_stages.append(reader_to_stage[stage])
            else:
                # Keep processing stages as-is
                execution_stages.append(stage)

        # Apply optimizations (fusion, etc.)
        optimized_stages, fusion_info = self.planner._fuse_stages(execution_stages)

        # Build execution graph
        graph = self.planner._build_graph(optimized_stages)

        # Create execution plan
        plan = ExecutionPlan(
            pipeline_name=spec.name,
            stages=optimized_stages,
            initial_tasks=all_file_groups,
            graph=graph,
            fusion_info=fusion_info,
        )

        return plan


# Convenience function for running pipelines
def run_pipeline(spec: PipelineSpec, executor_config: dict[str, Any] | None = None) -> list[Task]:
    """Run a pipeline specification.

    This is a convenience function that creates a runner and executes the spec.

    Args:
        spec: Pipeline specification to run
        executor_config: Optional executor configuration

    Returns:
        List of output tasks
    """
    runner = PipelineRunner(executor_config)
    return runner.run(spec)
