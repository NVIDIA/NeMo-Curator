"""Pipeline planning and optimization."""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from ray_curator.stages.base import ProcessingStage
from ray_curator.data import Task

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Optimized execution plan for a pipeline."""

    pipeline_name: str
    stages: list[ProcessingStage]
    graph: dict[str, Any]
    fusion_info: dict[str, list[str]]  # Maps fused stage names to original stage names
    decomposition_info: dict[str, list[str]]  # Maps composite stage names to execution stage names

    def describe(self) -> str:
        """Get a description of the execution plan."""
        lines = [
            f"Execution Plan for: {self.pipeline_name}",
            f"Optimized stages: {len(self.stages)}",
            "",
        ]

        if self.decomposition_info:
            lines.append("Composite stage decompositions:")
            for composite_name, execution_names in self.decomposition_info.items():
                lines.append(f"  - {composite_name} -> {', '.join(execution_names)}")
            lines.append("")

        if self.fusion_info:
            lines.append("Stage fusions:")
            for fused_name, original_names in self.fusion_info.items():
                lines.append(f"  - {fused_name} <- {', '.join(original_names)}")
            lines.append("")

        lines.append("Execution order:")
        for i, stage in enumerate(self.stages, 1):
            stage_type = "SOURCE" if i == 1 and stage.is_source_stage else "PROCESSING"
            lines.append(f"  {i}. [{stage_type}] {stage.name}")

        return "\n".join(lines)


class PipelinePlanner:
    """Plans and optimizes pipeline execution."""

    def __init__(self, enable_fusion: bool = False):
        """Initialize the pipeline planner.

        Args:
            enable_fusion: Whether to enable stage fusion optimization
        """
        self.enable_fusion = enable_fusion

    def plan(self, pipeline: "Pipeline") -> ExecutionPlan:
        """Create an optimized execution plan.

        Args:
            pipeline: Pipeline to plan

        Returns:
            Optimized execution plan
        """
        logger.info(f"Planning pipeline: {pipeline.name}")

        # 1. Validate pipeline has stages
        if not pipeline.stages:
            raise ValueError(f"Pipeline '{pipeline.name}' has no stages")

        # 2. Decompose composite stages into execution stages
        execution_stages, decomposition_info = self._decompose_stages(pipeline.stages)

        # 3. Validate first stage is a source stage
        if not execution_stages:
            raise ValueError("No execution stages after decomposition")
        
        first_stage = execution_stages[0]
        if not first_stage.is_source_stage:
            raise ValueError(
                f"First stage '{first_stage.name}' is not a source stage. "
                "Pipeline must start with a stage that creates tasks (e.g., reader, downloader). "
                "Source stages must have 'is_source_stage' property set to True."
            )

        # 4. Apply fusion optimization if enabled
        if self.enable_fusion:
            fused_stages, fusion_info = self._fuse_stages(execution_stages)
        else:
            fused_stages = execution_stages
            fusion_info = {}

        # 5. Create execution graph
        execution_graph = self._build_graph(fused_stages)

        plan = ExecutionPlan(
            pipeline_name=pipeline.name,
            stages=fused_stages,
            graph=execution_graph,
            fusion_info=fusion_info,
            decomposition_info=decomposition_info,
        )

        logger.info(
            f"Created execution plan with {len(fused_stages)} stages "
            f"(original: {len(pipeline.stages)}, decomposed: {len(execution_stages)})"
        )

        return plan

    def _fuse_stages(self, stages: list[ProcessingStage]) -> tuple[list[ProcessingStage], dict[str, list[str]]]:
        """Fuse compatible stages for efficiency.

        Args:
            stages: List of stages to potentially fuse

        Returns:
            Tuple of (fused stages list, fusion info dict)
        """
        # TODO: Implement stage fusion logic
        # For now, return stages as-is without any fusion
        logger.info("Stage fusion is not yet implemented - returning stages unchanged")
        return stages, {}

    def _can_fuse(self, stage1: ProcessingStage, stage2: ProcessingStage) -> bool:
        """Check if two stages can be fused.

        Args:
            stage1: First stage
            stage2: Second stage

        Returns:
            True if stages can be fused
        """
        # TODO: Implement fusion compatibility checks
        # Placeholder implementation - always return False
        return False

    def _create_fused_stage(self, stage1: ProcessingStage, stage2: ProcessingStage) -> ProcessingStage:
        """Create a fused stage from two compatible stages.

        Args:
            stage1: First stage
            stage2: Second stage

        Returns:
            Fused stage
        """
        # TODO: Implement fused stage creation
        # This is a placeholder that should never be called with current _can_fuse implementation
        raise NotImplementedError("Stage fusion is not yet implemented")

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
                "type": stage.__class__.__name__,
                "requires_gpu": stage.requires_gpu,
                "cpu_cores": stage.cpu_cores,
            }
            graph["nodes"].append(node)

        # Create edges (simple linear flow for now)
        for i in range(len(stages) - 1):
            edge = {"from": i, "to": i + 1}
            graph["edges"].append(edge)

        return graph

    def _decompose_stages(self, stages: list[ProcessingStage]) -> tuple[list[ProcessingStage], dict[str, list[str]]]:
        """Decompose composite stages into execution stages.
        
        Args:
            stages: List of stages that may include composite stages
            
        Returns:
            Tuple of (execution stages, decomposition info dict)
        """
        execution_stages = []
        decomposition_info = {}
        
        for stage in stages:
            # Get the decomposed stages (returns [self] for regular stages)
            sub_stages = stage.decompose()
            
            if len(sub_stages) > 1:
                # This was a composite stage
                logger.info(f"Decomposing composite stage: {stage.name}")
                
                # Validate that decomposed stages are not composite
                for sub_stage in sub_stages:
                    if len(sub_stage.decompose()) > 1:
                        raise ValueError(
                            f"Composite stage '{stage.name}' decomposed into another "
                            f"composite stage '{sub_stage.name}'. Nested composition "
                            "is not supported."
                        )
                
                execution_stages.extend(sub_stages)
                decomposition_info[stage.name] = [s.name for s in sub_stages]
                logger.info(f"Expanded '{stage.name}' into {len(sub_stages)} execution stages")
            else:
                # Regular stage, add as-is
                execution_stages.append(stage)
        
        return execution_stages, decomposition_info
