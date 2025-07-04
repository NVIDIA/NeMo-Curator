import json
import re
from pathlib import Path
from typing import Any

import pytest

from ray_curator.backends.base import BaseExecutor
from ray_curator.backends.experimental.ray_data import RayDataExecutor
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.tasks import FileGroupTask

from .utils import (
    EXPECTED_NUM_STAGES,
    TOTAL_DOCUMENTS,
    capture_logs,
    create_test_data,
    create_test_pipeline,
)


@pytest.mark.parametrize(
    "setup_pipeline",
    [
        (XennaExecutor, {}),
        (RayDataExecutor, {}),
    ],
    indirect=True,
)
class TestBackends:
    NUM_TEST_FILES = 3
    EXPECTED_OUTPUT_TASKS = EXPECTED_OUTPUT_FILES = TOTAL_DOCUMENTS  # After split_into_rows stage

    # Class attributes for shared test data
    # These are set by the setup_pipeline fixture
    backend_cls: BaseExecutor | None = None
    config: dict[str, Any] | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    output_tasks: list[FileGroupTask] | None = None
    all_logs: str = ""

    @pytest.fixture(scope="class", autouse=True)
    def setup_pipeline(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
        """Set up pipeline execution once per backend configuration."""
        # Get the backend class and config from the parametrized values
        backend_cls, config = request.param

        # Store as class attributes using request.cls (proper way for class-scoped fixtures)
        request.cls.backend_cls = backend_cls  # type: ignore[reportOptionalMemberAccess]
        request.cls.config = config  # type: ignore[reportOptionalMemberAccess]

        # Create fresh directories using tmp_path_factory for class-scoped fixture
        tmp_path = tmp_path_factory.mktemp("test_data")
        request.cls.input_dir = tmp_path / "input"  # type: ignore[reportOptionalMemberAccess]
        request.cls.output_dir = tmp_path / "output"  # type: ignore[reportOptionalMemberAccess]

        # Create test data and pipeline
        create_test_data(request.cls.input_dir, num_files=self.NUM_TEST_FILES)  # type: ignore[reportOptionalMemberAccess]
        pipeline = create_test_pipeline(request.cls.input_dir, request.cls.output_dir)  # type: ignore[reportOptionalMemberAccess]

        # Execute pipeline with comprehensive logging capture
        executor = backend_cls(**config)
        with capture_logs() as log_buffer:
            request.cls.output_tasks = pipeline.run(executor)  # type: ignore[reportOptionalMemberAccess]
            # Store logs for backend-specific tests
            request.cls.all_logs = log_buffer.getvalue()  # type: ignore[reportOptionalMemberAccess]

    def test_output_files(self):
        """Test that the correct number of output files are created with expected content."""
        assert self.output_dir is not None, "Output directory should be set by fixture"

        # Check file count
        output_files = list(self.output_dir.glob("*.jsonl"))
        assert len(output_files) == self.EXPECTED_OUTPUT_FILES, "Mismatch in number of output files"

        # Check file contents
        for file in output_files:
            with open(file) as f:
                for line in f:
                    data = json.loads(line)
                    assert set(data.keys()) == {"id", "doc_length", "text"}, "Mismatch in output file contents"

    def test_output_tasks(self):
        """Test that output tasks have the correct count, types, and properties."""
        assert self.output_tasks is not None, "Expected output tasks"

        # Check task count
        assert len(self.output_tasks) == self.EXPECTED_OUTPUT_TASKS, "Mismatch in number of output tasks"

        # Check all tasks are of type FileGroupTask
        assert all(isinstance(task, FileGroupTask) for task in self.output_tasks), "Mismatch in task types"

        # Check all task_ids are unique
        assert len({task.task_id for task in self.output_tasks}) == self.EXPECTED_OUTPUT_TASKS, (
            "Mismatch in number of task ids"
        )

        # Check all dataset names are the same
        assert all(task.dataset_name == self.output_tasks[0].dataset_name for task in self.output_tasks), (
            "Mismatch in dataset names"
        )

    def test_perf_stats(self):
        """Test that performance statistics are correctly recorded for all stages."""
        # Check content of stage perf stats
        assert self.output_tasks is not None, "Expected output tasks"
        expected_stage_names = ["jsonl_reader", "add_length", "split_into_rows", "jsonl_writer"]
        for task_idx, task in enumerate(self.output_tasks):
            assert len(task._stage_perf) == EXPECTED_NUM_STAGES, "Mismatch in number of stage perf stats"
            # Make sure stage names match
            for stage_idx, perf_stats in enumerate(task._stage_perf):
                assert perf_stats.stage_name == expected_stage_names[stage_idx], (
                    f"Mismatch in stage name for stage {stage_idx} within task {task_idx}"
                )
                # Process time should be greater than idle time
                assert perf_stats.process_time > 0, "Process time should be non-zero for all stages"
            assert task._stage_perf[1].num_items_processed == task._stage_perf[2].num_items_processed, (
                "Mismatch in number of items processed by add_length and split_into_rows"
            )
            # Because we split df into a single row each, the number of items processed by jsonl_writer should be only 1 i.e 1 row
            assert task._stage_perf[3].num_items_processed == 1, (
                "Mismatch in number of items processed by jsonl_writer"
            )

    def test_ray_data_execution_plan(self):
        """Test that Ray Data creates the expected execution plan with correct stage organization."""
        if self.backend_cls != RayDataExecutor:
            pytest.skip("Execution plan test only applies to RayDataExecutor")

        # Look for execution plan in logs with multiple possible patterns
        matches = re.findall(r"Execution plan of Dataset.*?:\s*(.+)", self.all_logs, re.MULTILINE)
        # Take the last execution plan (most recent)
        execution_plan = matches[-1]
        # Split by " -> " to get individual stages
        stages = execution_plan.split(" -> ")
        execution_plan_stages = [stage.strip() for stage in stages]
        expected_stages = [
            "InputDataBuffer[Input]",
            "TaskPoolMapOperator[MapBatches(FilePartitioningStage)]",
            "TaskPoolMapOperator[StreamingRepartition->MapBatches(JsonlReaderStage)->MapBatches(AddLengthStage)]",
            "TaskPoolMapOperator[MapBatches(SplitIntoRowsStage)]",
            "TaskPoolMapOperator[StreamingRepartition->MapBatches(JsonlWriter)]",
        ]

        assert execution_plan_stages == expected_stages, (
            f"Expected execution plan stages: {expected_stages}, got: {execution_plan_stages}"
        )
