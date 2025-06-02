"""Pipeline definition for composing processing stages."""

from typing import Any

from loguru import logger

from ray_curator.readers.base import Reader
from ray_curator.stages.base import ProcessingStage


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
        self.readers: list[Reader] = []
        self._config: dict[str, Any] = {}

    def add_reader(self, reader: Reader) -> "Pipeline":
        """Add a reader to the pipeline.

        Readers are special components that create file groups during planning.
        They must be added before any processing stages.

        Args:
            reader: Reader configuration

        Returns:
            Self for method chaining
        """
        if not isinstance(reader, Reader):
            raise TypeError(f"Reader must be a Reader instance, got {type(reader)}")

        if self.stages:
            raise ValueError("Readers must be added before processing stages")

        self.readers.append(reader)
        logger.info(f"Added reader to pipeline '{self.name}'")
        return self

    def add_stage(self, stage: ProcessingStage | Reader) -> "Pipeline":
        """Add a stage to the pipeline.

        This method accepts both ProcessingStage and Reader for backward compatibility.
        If a Reader is passed, it will be added via add_reader.

        Args:
            stage: Processing stage or reader to add

        Returns:
            Self for method chaining
        """
        if isinstance(stage, Reader):
            return self.add_reader(stage)

        if not isinstance(stage, ProcessingStage):
            raise TypeError(f"Stage must be a ProcessingStage or Reader, got {type(stage)}")

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
        if not self.stages and not self.readers:
            raise ValueError(f"Pipeline '{self.name}' has no stages or readers")

        # Check stage names are unique
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("Pipeline contains duplicate stage names")

        # Validate each stage
        for i, stage in enumerate(self.stages):
            try:
                # Basic validation - stages should have required properties
                _ = stage.name
                _ = stage.stage_type
                _ = stage.process
            except Exception as e:
                raise ValueError(f"Invalid stage at position {i}: {e}")

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
        components = []

        if self.readers:
            reader_info = f"{len(self.readers)} reader(s)"
            components.append(reader_info)

        if self.stages:
            stage_info = ", ".join([f"{s.name}({s.stage_type.value})" for s in self.stages])
            components.append(f"stages=[{stage_info}]")

        return f"Pipeline(name='{self.name}', {', '.join(components)})"

    def describe(self) -> str:
        """Get a detailed description of the pipeline."""
        lines = [f"Pipeline: {self.name}", f"Description: {self.description or 'No description provided'}", ""]

        if self.readers:
            lines.append(f"Readers: {len(self.readers)}")
            for i, reader in enumerate(self.readers, 1):
                lines.append(f"  {i}. {reader.__class__.__name__}")
            lines.append("")

        if self.stages:
            lines.append(f"Processing Stages: {len(self.stages)}")
            for i, stage in enumerate(self.stages, 1):
                lines.append(f"  {i}. {stage.name} ({stage.stage_type.value})")
                if stage.requires_gpu:
                    lines.append(f"     - Requires GPU: {stage.gpu_memory_gb}GB")
                lines.append(f"     - CPU cores: {stage.cpu_cores}")
                if stage.can_fuse_with:
                    fusable = ", ".join([t.value for t in stage.can_fuse_with])
                    lines.append(f"     - Can fuse with: {fusable}")

        return "\n".join(lines)
