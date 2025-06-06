"""Pipeline definition for composing processing stages."""

import logging
from typing import Any

from ray_curator.stages.base import ProcessingStage

logger = logging.getLogger(__name__)


class Pipeline:
    """User-facing pipeline definition for composing processing stages."""

    def __init__(self, name: str, description: str | None = None):
        """Initialize a new pipeline.

        Args:
            name: Name of the pipeline
            description: Optional description of what the pipeline does
        """
        self.name = name
        self.description = description
        self.stages: list[ProcessingStage] = []
        self._config: dict[str, Any] = {}

    def add_stage(self, stage: ProcessingStage) -> "Pipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Processing stage to add

        Returns:
            Self for method chaining
        """
        if not isinstance(stage, ProcessingStage):
            raise TypeError(f"Stage must be a ProcessingStage, got {type(stage)}")

        self.stages.append(stage)
        logger.info(f"Added stage '{stage.name}' to pipeline '{self.name}'")
        return self

    def set_config(self, **kwargs) -> "Pipeline":
        """Set configuration parameters for the pipeline.

        Args:
            **kwargs: Configuration parameters

        Returns:
            Self for method chaining
        """
        self._config.update(kwargs)
        return self

    def get_config(self) -> dict[str, Any]:
        """Get pipeline configuration.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def validate(self) -> bool:
        """Validate the pipeline configuration.

        Returns:
            True if valid, raises exception otherwise
        """
        if not self.stages:
            raise ValueError(f"Pipeline '{self.name}' has no stages")

        # Check stage names are unique
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("Pipeline contains duplicate stage names")

        # Validate each stage
        for i, stage in enumerate(self.stages):
            try:
                # Basic validation - stages should have required properties
                _ = stage.name
                _ = stage.process
            except Exception as e:
                raise ValueError(f"Invalid stage at position {i}: {e}")

        # Validate first stage is a source stage
        # This will be checked more thoroughly in the planner
        first_stage = self.stages[0]
        if not first_stage.is_source_stage:
            logger.warning(
                f"First stage '{first_stage.name}' is not marked as a source stage. "
                "The pipeline planner will validate this more strictly."
            )

        return True

    def build(self) -> "ExecutionPlan":
        """Build an execution plan from the pipeline.

        Returns:
            Optimized execution plan
        """
        from .planner import PipelinePlanner

        # Validate before building
        self.validate()

        # Use planner to create execution plan
        planner = PipelinePlanner()
        return planner.plan(self)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        stage_info = ", ".join([f"{s.name}({s.__class__.__name__})" for s in self.stages])
        return f"Pipeline(name='{self.name}', stages=[{stage_info}])"

    def describe(self) -> str:
        """Get a detailed description of the pipeline."""
        lines = [
            f"Pipeline: {self.name}",
            f"Description: {self.description or 'No description provided'}",
            f"Stages: {len(self.stages)}",
            ""
        ]

        for i, stage in enumerate(self.stages, 1):
            stage_type = "SOURCE" if i == 1 and stage.is_source_stage else "PROCESSING"
            lines.append(f"  {i}. [{stage_type}] {stage.name} ({stage.__class__.__name__})")
            if stage.requires_gpu:
                lines.append(f"     - Requires GPU: {stage.gpu_memory_gb}GB")
            lines.append(f"     - CPU cores: {stage.cpu_cores}")

        return "\n".join(lines)
