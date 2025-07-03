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

    def __init__(self, name: str = "ConcreteProcessingStage", resources: Resources | None = None, batch_size: int = 1):
        super().__init__(name=name, resources=resources or Resources(cpus=2.0), batch_size=batch_size)

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
