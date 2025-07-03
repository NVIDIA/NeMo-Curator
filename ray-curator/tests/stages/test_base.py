"""Tests for base stage classes."""

from dataclasses import dataclass

from ray_curator.stages.base import ProcessingStage
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


@dataclass
class ConcreteProcessingStage(ProcessingStage[MockTask, MockTask]):
    """Concrete implementation of ProcessingStage for testing."""

    _name = "ConcreteProcessingStage"
    _resources = Resources(cpus=2.0)
    _batch_size = 2

    def process(self, task: MockTask) -> MockTask:
        return task

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []


class TestProcessingStageWith:
    """Test the with_ method for ProcessingStage."""

    def test_all(self):
        stage = ConcreteProcessingStage()
        assert stage.resources.cpus == 2.0

        # Test the simplified with_ method that modifies the instance directly
        stage.with_(resources=Resources(cpus=4.0))
        assert stage.resources.cpus == 4.0

        # Test with name override
        stage.with_(name="CustomStage")
        assert stage.name == "CustomStage"

    def test_batch_size_override(self):
        """Test overriding batch_size parameter."""
        stage = ConcreteProcessingStage()
        assert stage.batch_size == 2

        stage.with_(batch_size=5)
        assert stage.batch_size == 5

    def test_multiple_parameters(self):
        """Test overriding multiple parameters at once."""
        stage = ConcreteProcessingStage()
        new_resources = Resources(cpus=3.0)
        stage.with_(name="MultiParamStage", resources=new_resources, batch_size=10)

        assert stage.name == "MultiParamStage"
        assert stage.resources.cpus == 3.0
        assert stage.batch_size == 10

    def test_none_parameters_preserve_original(self):
        """Test that None parameters preserve original values."""
        stage = ConcreteProcessingStage()
        original_name = stage.name
        original_resources = stage.resources
        original_batch_size = stage.batch_size

        # Pass None for all parameters
        stage.with_(name=None, resources=None, batch_size=None)

        # Values should remain unchanged
        assert stage.name == original_name
        assert stage.resources == original_resources
        assert stage.batch_size == original_batch_size

    def test_chained_with_calls(self):
        """Test that with_ can be chained and returns self."""
        stage = ConcreteProcessingStage()

        # Chain multiple with_ calls
        result = stage.with_(name="ChainedStage").with_(batch_size=8).with_(resources=Resources(cpus=6.0))

        # Should return self
        assert result is stage
        assert stage.name == "ChainedStage"
        assert stage.batch_size == 8
        assert stage.resources.cpus == 6.0
