"""Tests for base stage classes."""

import pytest

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Task


class MockTask(Task[dict]):
    """Mock task for testing."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        self.data = data or {}

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class ConcreteProcessingStage(ProcessingStage[MockTask, MockTask]):
    """Concrete implementation of ProcessingStage for testing."""

    def __init__(self, name: str = "test_stage"):
        self._name = name
        self._resources = Resources(cpus=1.0)
        self._batch_size = 1

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def resources(self) -> Resources:
        return self._resources

    @resources.setter
    def resources(self, value: Resources) -> None:
        self._resources = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = value

    def process(self, task: MockTask) -> MockTask:
        return task

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []


class ConcreteCompositeStage(CompositeStage[MockTask, MockTask]):
    """Concrete implementation of CompositeStage for testing."""

    def __init__(self, name: str = "composite_stage"):
        self._name = name
        self._stages = [ConcreteProcessingStage("stage1"), ConcreteProcessingStage("stage2")]

    @property
    def name(self) -> str:
        return self._name

    def decompose(self) -> list[ProcessingStage]:
        return self._stages

    def process(self, task: MockTask) -> MockTask:
        return super().process(task)


class TestProcessingStageWith:
    """Test the with_ method for ProcessingStage."""

    def test_with_name_only(self):
        """Test with_ method with only name parameter."""
        stage = ConcreteProcessingStage("original_name")

        result = stage.with_(name="new_name")

        assert result is stage  # Should return self
        assert stage.name == "new_name"
        assert stage.resources.cpus == 1.0  # Should remain unchanged
        assert stage.batch_size == 1  # Should remain unchanged

    def test_with_resources_only(self):
        """Test with_ method with only resources parameter."""
        stage = ConcreteProcessingStage()
        new_resources = Resources(cpus=4.0, gpu_memory_gb=8.0)

        result = stage.with_(resources=new_resources)

        assert result is stage  # Should return self
        assert stage.name == "test_stage"  # Should remain unchanged
        assert stage.resources.cpus == 4.0
        assert stage.resources.gpu_memory_gb == 8.0
        assert stage.batch_size == 1  # Should remain unchanged

    def test_with_batch_size_only(self):
        """Test with_ method with only batch_size parameter."""
        stage = ConcreteProcessingStage()

        result = stage.with_(batch_size=10)

        assert result is stage  # Should return self
        assert stage.name == "test_stage"  # Should remain unchanged
        assert stage.resources.cpus == 1.0  # Should remain unchanged
        assert stage.batch_size == 10

    def test_with_multiple_parameters(self):
        """Test with_ method with multiple parameters."""
        stage = ConcreteProcessingStage("original_name")
        new_resources = Resources(cpus=2.0, gpus=1.0)

        result = stage.with_(name="new_name", resources=new_resources, batch_size=5)

        assert result is stage  # Should return self
        assert stage.name == "new_name"
        assert stage.resources.cpus == 2.0
        assert stage.resources.gpus == 1.0
        assert stage.batch_size == 5

    def test_with_none_parameters(self):
        """Test with_ method with None parameters (should not change values)."""
        stage = ConcreteProcessingStage("original_name")
        original_resources = stage.resources
        original_batch_size = stage.batch_size

        result = stage.with_(name=None, resources=None, batch_size=None)

        assert result is stage  # Should return self
        assert stage.name == "original_name"  # Should remain unchanged
        assert stage.resources is original_resources  # Should remain unchanged
        assert stage.batch_size == original_batch_size  # Should remain unchanged


class TestCompositeStageWith:
    """Test the with_ method for CompositeStage."""

    def test_with_valid_stage_config(self):
        """Test with_ method with valid stage configuration."""
        composite = ConcreteCompositeStage()
        stage1, stage2 = composite.decompose()

        # Verify initial state
        assert stage1.name == "stage1"
        assert stage1.resources.cpus == 1.0
        assert stage1.batch_size == 1
        assert stage2.name == "stage2"
        assert stage2.resources.cpus == 1.0
        assert stage2.batch_size == 1

        # Apply configuration changes
        result = composite.with_(
            {
                "stage1": {"name": "new_stage1", "batch_size": 5},
                "stage2": {"resources": Resources(cpus=4.0), "batch_size": 10},
            }
        )

        assert result is composite  # Should return self

        # Verify changes were applied
        assert stage1.name == "new_stage1"
        assert stage1.resources.cpus == 1.0  # Should remain unchanged
        assert stage1.batch_size == 5

        assert stage2.name == "stage2"  # Should remain unchanged
        assert stage2.resources.cpus == 4.0
        assert stage2.batch_size == 10

    def test_with_duplicate_stage_names(self):
        """Test with_ method with duplicate stage names (should raise ValueError)."""

        # Create a composite stage with duplicate names
        class DuplicateNameCompositeStage(CompositeStage[MockTask, MockTask]):
            def __init__(self):
                self._name = "duplicate_composite"
                self._stages = [ConcreteProcessingStage("duplicate_name"), ConcreteProcessingStage("duplicate_name")]

            @property
            def name(self) -> str:
                return self._name

            def decompose(self) -> list[ProcessingStage]:
                return self._stages

            def process(self, task: MockTask) -> MockTask:
                return super().process(task)

        composite = DuplicateNameCompositeStage()

        with pytest.raises(ValueError, match="All stages must have unique names"):
            composite.with_({"duplicate_name": {"batch_size": 5}})

    def test_with_nonexistent_stage_name(self):
        """Test with_ method with non-existent stage name."""
        composite = ConcreteCompositeStage()

        # This should raise a KeyError when stage name doesn't exist
        with pytest.raises(KeyError, match="nonexistent_stage"):
            composite.with_({"nonexistent_stage": {"batch_size": 5}})

    def test_with_empty_config(self):
        """Test with_ method with empty configuration."""
        composite = ConcreteCompositeStage()
        stage1, stage2 = composite.decompose()

        # Store original values
        original_name1 = stage1.name
        original_name2 = stage2.name
        original_batch_size1 = stage1.batch_size
        original_batch_size2 = stage2.batch_size

        result = composite.with_({})

        assert result is composite  # Should return self

        # Verify no changes were made
        assert stage1.name == original_name1
        assert stage2.name == original_name2
        assert stage1.batch_size == original_batch_size1
        assert stage2.batch_size == original_batch_size2
