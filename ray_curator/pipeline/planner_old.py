"""Pipeline planning and optimization."""

import logging
from dataclasses import dataclass
from typing import Any

from ray_curator.readers.base import FileGroupTask, Reader
from ray_curator.stages.base import ProcessingStage, StageType
from ray_curator.stages.readers import JsonlProcessingStage

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Optimized execution plan for a pipeline."""

    pipeline_name: str
    stages: list[ProcessingStage]
    initial_tasks: list[FileGroupTask]  # File groups created from readers
    graph: dict[str, Any]
    fusion_info: dict[str, list[str]]  # Maps fused stage names to original stage names

    def describe(self) -> str:
        """Get a description of the execution plan."""
        lines = [
            f"Execution Plan for: {self.pipeline_name}",
            f"Initial tasks: {len(self.initial_tasks)}",
            f"Optimized stages: {len(self.stages)}",
            "",
        ]

        if self.initial_tasks:
            lines.append("File groups to process:")
            for task in self.initial_tasks[:5]:  # Show first 5
                lines.append(f"  - {task.task_id}: {len(task.file_paths)} files")
            if len(self.initial_tasks) > 5:
                lines.append(f"  ... and {len(self.initial_tasks) - 5} more")
            lines.append("")

        if self.fusion_info:
            lines.append("Stage fusions:")
            for fused_name, original_names in self.fusion_info.items():
                lines.append(f"  - {fused_name} <- {', '.join(original_names)}")
            lines.append("")

        lines.append("Execution order:")
        for i, stage in enumerate(self.stages, 1):
            lines.append(f"  {i}. {stage.name}")

        return "\n".join(lines)


class PipelinePlanner:
    """Plans and optimizes pipeline execution."""

    def __init__(self):
        """Initialize the planner."""
        # Map reader stage names to their processing stage classes
        self._reader_stage_map = {
            "jsonl_processing_stage": JsonlProcessingStage,
            "parquet_processing_stage": None,
        }

    def plan(self, pipeline: "Pipeline") -> ExecutionPlan:
        """Create an optimized execution plan.
        Args:
            pipeline: Pipeline to plan
        Returns:
            Optimized execution plan
        """
        logger.info(f"Planning pipeline: {pipeline.name}")

        # 1. Process readers to create initial tasks and stages
        initial_tasks, reader_stages = self._process_readers(pipeline.readers)

        # 2. Combine reader stages with user-defined stages
        all_stages = reader_stages + pipeline.stages

        # 3. Identify fusion opportunities
        fused_stages, fusion_info = self._fuse_stages(all_stages)

        # 4. Create execution graph
        execution_graph = self._build_graph(fused_stages)

        plan = ExecutionPlan(
            pipeline_name=pipeline.name,
            stages=fused_stages,
            initial_tasks=initial_tasks,
            graph=execution_graph,
            fusion_info=fusion_info,
        )

        logger.info(
            f"Created execution plan with {len(initial_tasks)} initial tasks "
            f"and {len(fused_stages)} stages (original: {len(all_stages)})"
        )

        return plan

    def _process_readers(self, readers: list[Reader]) -> tuple[list[FileGroupTask], list[ProcessingStage]]:
        """Process readers to create file groups and corresponding stages.
        Args:
            readers: List of reader configurations
        Returns:
            Tuple of (initial tasks, reader stages)
        """
        all_tasks = []
        reader_stages = []

        for reader in readers:
            # Create file groups
            file_groups = reader.create_file_groups()
            all_tasks.extend(file_groups)

            # Create corresponding processing stage
            stage_name = reader.get_reader_stage_name()
            stage_class = self._reader_stage_map.get(stage_name)

            if stage_class:
                # Create instance of the processing stage
                stage = stage_class()
                reader_stages.append(stage)
                logger.info(f"Created {len(file_groups)} file groups for {reader.__class__.__name__}")
            else:
                logger.warning(f"Unknown reader stage: {stage_name}")

        return all_tasks, reader_stages

    def _fuse_stages(self, stages: list[ProcessingStage]) -> tuple[list[ProcessingStage], dict[str, list[str]]]:
        """Fuse compatible stages for efficiency.
        Args:
            stages: List of stages to potentially fuse
        Returns:
            Tuple of (fused stages list, fusion info dict)
        """
        if len(stages) < 2:
            return stages, {}

        fused_stages = []
        fusion_info = {}
        i = 0

        while i < len(stages):
            current_stage = stages[i]

            # Check if we can fuse with the next stage
            if i + 1 < len(stages):
                next_stage = stages[i + 1]

                if self._can_fuse(current_stage, next_stage):
                    # Create fused stage
                    fused = self._create_fused_stage(current_stage, next_stage)
                    fused_stages.append(fused)
                    fusion_info[fused.name] = [current_stage.name, next_stage.name]
                    logger.info(f"Fused stages: {current_stage.name} + {next_stage.name}")
                    i += 2  # Skip both stages
                    continue

            # No fusion possible, add stage as-is
            fused_stages.append(current_stage)
            i += 1

        return fused_stages, fusion_info

    def _can_fuse(self, stage1: ProcessingStage, stage2: ProcessingStage) -> bool:
        """Check if two stages can be fused.
        Args:
            stage1: First stage
            stage2: Second stage
        Returns:
            True if stages can be fused
        """
        # Don't fuse reader stages
        if stage1.stage_type == StageType.READER or stage2.stage_type == StageType.READER:
            return False

        # Check if stage1 declares it can fuse with stage2's type
        if stage2.stage_type in stage1.can_fuse_with:
            return True

        # Check if stage2 declares it can fuse with stage1's type
        if stage1.stage_type in stage2.can_fuse_with:
            return True

        return False

    def _create_fused_stage(self, stage1: ProcessingStage, stage2: ProcessingStage) -> ProcessingStage:
        """Create a fused stage from two compatible stages.
        Args:
            stage1: First stage
            stage2: Second stage
        Returns:
            Fused stage
        """

        from ray_curator.data import Task
        from ray_curator.stages.base import ProcessingStage, StageType

        class FusedStage(ProcessingStage):
            """Dynamically created fused stage."""

            def __init__(self, s1: ProcessingStage, s2: ProcessingStage):
                self.stage1 = s1
                self.stage2 = s2

            @property
            def name(self) -> str:
                return f"fused_{self.stage1.name}_{self.stage2.name}"

            @property
            def stage_type(self) -> StageType:
                return StageType.FUSED

            @property
            def requires_gpu(self) -> bool:
                return self.stage1.requires_gpu or self.stage2.requires_gpu

            @property
            def gpu_memory_gb(self) -> float:
                return max(self.stage1.gpu_memory_gb, self.stage2.gpu_memory_gb)

            @property
            def cpu_cores(self) -> float:
                return max(self.stage1.cpu_cores, self.stage2.cpu_cores)

            def process(self, task: Task) -> Task | list[Task] | None:
                # Process through first stage
                intermediate = self.stage1.process(task)
                if intermediate is None:
                    return None

                # Handle case where first stage returns multiple tasks
                if isinstance(intermediate, list):
                    results = []
                    for t in intermediate:
                        result = self.stage2.process(t)
                        if result is not None:
                            if isinstance(result, list):
                                results.extend(result)
                            else:
                                results.append(result)
                    return results if results else None
                else:
                    # Single task
                    return self.stage2.process(intermediate)

            def setup(self) -> None:
                self.stage1.setup()
                self.stage2.setup()

            def teardown(self) -> None:
                self.stage2.teardown()
                self.stage1.teardown()

        return FusedStage(stage1, stage2)

    def _build_graph(self, stages: list[ProcessingStage]) -> dict[str, Any]:
        """Build execution graph.
        Args:
            stages: List of stages
        Returns:
            Execution graph as dictionary
        """
        graph = {"nodes": [], "edges": []}

        # Create nodes for each stage
        for i, stage in enumerate(stages):
            node = {
                "id": i,
                "name": stage.name,
                "type": stage.stage_type.value,
                "requires_gpu": stage.requires_gpu,
                "cpu_cores": stage.cpu_cores,
            }
            graph["nodes"].append(node)

        # Create edges (simple linear flow for now)
        for i in range(len(stages) - 1):
            edge = {"from": i, "to": i + 1}
            graph["edges"].append(edge)

        return graph
