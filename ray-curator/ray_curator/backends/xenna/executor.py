from typing import Any

from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.utils.verbosity import VerbosityLevel
from loguru import logger

from ray_curator.backends.base import BaseExecutor
from ray_curator.backends.xenna.adapter import create_named_xenna_stage_adapter
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import EmptyTask, Task


class XennaExecutor(BaseExecutor):
    """Executor that runs pipelines using Cosmos-Xenna.
    This executor provides integration between the nemo-curator pipeline framework
    and the Cosmos-Xenna execution engine for distributed processing.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the executor.

        Args:
            config (dict[str, Any], optional): Configuration dictionary with options like:
                - logging_interval: Seconds between status logs (default: 60)
                - ignore_failures: Whether to continue on failures (default: False)
                - max_workers_per_stage: Max workers per stage (default: None)
                - execution_mode: 'streaming' or 'batch' (default: 'streaming')
                - cpu_allocation_percentage: CPU allocation ratio (default: 0.95)
                - autoscale_interval_s: Auto-scaling interval (default: 180)
        """
        super().__init__(config)
        self._default_pipeline_config = {
            "logging_interval": 60,
            "ignore_failures": False,
            "execution_mode": "streaming",
            "cpu_allocation_percentage": 0.95,
            "autoscale_interval_s": 180,
        }

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        """Execute the pipeline using Cosmos-Xenna.

        Args:
            stages (list[ProcessingStage]): The stages to run
            initial_tasks (list[Task], optional): The initial tasks to run. Empty list of Task is used if not provided.

        Returns:
            list[Task]: List of output tasks from the pipeline
        """
        # Convert stages to Xenna stage specs
        stage_specs = []

        # Initialize with initial tasks if provided, otherwise start with EmptyTask
        initial_tasks = initial_tasks if initial_tasks else [EmptyTask]

        for stage in stages:
            # Get stage configuration
            stage_config = stage.xenna_stage_spec()

            # Create Xenna stage adapter with the original stage's name
            xenna_stage = create_named_xenna_stage_adapter(
                stage=stage,
            )

            # Create stage spec with configuration from stage
            stage_spec = pipelines_v1.StageSpec(
                stage=xenna_stage,
                num_workers=stage_config.get("num_workers"),
                num_workers_per_node=stage_config.get("num_workers_per_node"),
                num_setup_attempts_python=stage_config.get("num_setup_attempts_python"),
                num_run_attempts_python=stage_config.get("num_run_attempts_python"),
                ignore_failures=stage_config.get("ignore_failures"),
                reset_workers_on_failure=stage_config.get("reset_workers_on_failure"),
                slots_per_actor=stage_config.get("slots_per_actor"),
                worker_max_lifetime_m=stage_config.get("worker_max_lifetime_m"),
                worker_restart_interval_m=stage_config.get("worker_restart_interval_m"),
                max_setup_failure_percentage=stage_config.get("max_setup_failure_percentage"),
            )

            stage_specs.append(stage_spec)

        # Determine execution mode
        exec_mode = pipelines_v1.ExecutionMode.STREAMING
        if self._get_pipeline_config("execution_mode") == "batch":
            exec_mode = pipelines_v1.ExecutionMode.BATCH

        # Create streaming-specific configuration
        streaming_config = None
        if exec_mode == pipelines_v1.ExecutionMode.STREAMING:
            streaming_config = pipelines_v1.StreamingSpecificSpec(
                autoscale_interval_s=self._get_pipeline_config("autoscale_interval_s"),
                autoscaler_verbosity_level=VerbosityLevel.INFO,  # TODO: Move this to pipeline config
                executor_verbosity_level=VerbosityLevel.INFO,
            )

        # Create pipeline configuration
        pipeline_config = pipelines_v1.PipelineConfig(
            execution_mode=exec_mode,
            logging_interval_s=self._get_pipeline_config("logging_interval"),
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
            ignore_failures=self._get_pipeline_config("ignore_failures"),
            cpu_allocation_percentage=self._get_pipeline_config("cpu_allocation_percentage"),
            mode_specific=streaming_config,
            actor_pool_verbosity_level=VerbosityLevel.INFO,  # TODO: Move this to pipeline config
            monitoring_verbosity_level=VerbosityLevel.INFO,
        )

        # Create pipeline specification
        pipeline_spec = pipelines_v1.PipelineSpec(input_data=initial_tasks, stages=stage_specs, config=pipeline_config)

        # Log pipeline configuration
        logger.info(f"Execution mode: {exec_mode.name}")

        try:
            # Run the pipeline
            results = pipelines_v1.run_pipeline(pipeline_spec)
            logger.info(f"Pipeline completed successfully with {len(results) if results else 0} output tasks")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

        return results if results else []

    def _get_pipeline_config(self, key: str) -> Any:  # noqa: ANN401
        """Get configuration value with fallback to defaults."""
        return self.config.get(key, self._default_pipeline_config.get(key))
