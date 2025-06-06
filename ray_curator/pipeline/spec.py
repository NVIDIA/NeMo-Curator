"""Pipeline specification for defining data processing workflows."""

from dataclasses import dataclass, field
from typing import Any

from ray_curator.stages.readers.base import Reader
from ray_curator.stages.base import ProcessingStage


@dataclass
class PipelineSpec:
    """User-facing pipeline specification.

    This represents the logical pipeline as defined by the user,
    before it's transformed into an execution plan.

    Example:
        >>> spec = PipelineSpec(
        ...     stages=[
        ...         JsonlReader("data/*.jsonl", files_per_partition=5),
        ...         TextLengthFilterStage(min_length=100),
        ...         HtmlTextExtractorStage(),
        ...     ]
        ... )
        >>> pipeline.run(spec)
    """

    stages: list[Reader | ProcessingStage]
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage")

        # Auto-generate name if not provided
        if not self.name:
            if isinstance(self.stages[0], Reader):
                reader_type = self.stages[0].__class__.__name__.replace("Reader", "").lower()
                self.name = f"{reader_type}_pipeline"
            else:
                self.name = "data_pipeline"

    def validate(self) -> bool:
        """Validate the pipeline specification.

        Returns:
            True if valid

        Raises:
            ValueError: If the pipeline is invalid
        """
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage")

        # Check that all items are either Readers or ProcessingStages
        for i, stage in enumerate(self.stages):
            if not isinstance(stage, (Reader, ProcessingStage)):
                raise TypeError(f"Stage at position {i} must be a Reader or ProcessingStage, got {type(stage)}")

        # If pipeline starts with ProcessingStage, warn that no input data is defined
        if not any(isinstance(s, Reader) for s in self.stages):
            raise ValueError(
                "Pipeline must contain at least one Reader to define input data. "
                "Readers can be placed anywhere in the pipeline."
            )

        return True

    def get_readers(self) -> list[Reader]:
        """Get all readers from the pipeline."""
        return [s for s in self.stages if isinstance(s, Reader)]

    def get_processing_stages(self) -> list[ProcessingStage]:
        """Get all processing stages from the pipeline."""
        return [s for s in self.stages if isinstance(s, ProcessingStage)]

    def describe(self) -> str:
        """Get a human-readable description of the pipeline."""
        lines = [
            f"Pipeline: {self.name}",
            f"Description: {self.description or 'No description provided'}",
            f"Stages: {len(self.stages)}",
            "",
        ]

        for i, stage in enumerate(self.stages, 1):
            if isinstance(stage, Reader):
                stage_type = "Reader"
                stage_name = stage.__class__.__name__
            else:
                stage_type = stage.__class__.__name__
                stage_name = stage.name

            lines.append(f"  {i}. [{stage_type}] {stage_name}")

            # Add details for readers
            if isinstance(stage, Reader) and hasattr(stage, "file_paths"):
                lines.append(f"     - Input: {stage.file_paths}")
                if hasattr(stage, "files_per_partition") and stage.files_per_partition:
                    lines.append(f"     - Files per partition: {stage.files_per_partition}")
                elif hasattr(stage, "blocksize") and stage.blocksize:
                    lines.append(f"     - Block size: {stage.blocksize}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the pipeline spec."""
        stage_names = []
        for stage in self.stages:
            if isinstance(stage, Reader):
                stage_names.append(stage.__class__.__name__)
            else:
                stage_names.append(stage.name)

        return f"PipelineSpec(name='{self.name}', stages=[{', '.join(stage_names)}])"
