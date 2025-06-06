# Ray-Curator Pipeline Framework

The ray-curator pipeline framework provides a flexible and scalable way to process text data through a series of stages. All data is processed as dataframes for efficiency.

## Overview

The framework consists of several key components:

1. **Data Structures**
   - `Task`: Represents a batch of documents as a dataframe (PyArrow or Pandas)

2. **Stages**
   - `ProcessingStage`: Base class for all processing stages
   - Built-in stages: readers, downloaders, extractors, filters, transformers

3. **Pipeline**
   - `Pipeline`: User-facing API for composing stages
   - `PipelinePlanner`: Optimizes pipeline execution (stage fusion, etc.)
   - `ExecutionPlan`: Optimized execution plan

4. **Executors**
   - `XennaExecutor`: Executes pipelines using Cosmos-Xenna or Ray

## Data Model

Documents are processed in dataframes with these required columns:
- `id`: Unique document identifier
- `content`: Text content
- `metadata`: Dictionary with additional metadata (optional but recommended)

Additional columns can be added by stages during processing.

## Usage Example

```python
from ray_curator import (
    Pipeline,
    JsonlReaderStage,
    HtmlExtractorStage,
    DocumentFilterStage,
    XennaExecutor
)
from ray_curator.stages.filters import length_filter

# Create pipeline
pipeline = Pipeline("my_pipeline")
pipeline.add_stage(
    JsonlReaderStage(["data.jsonl"], batch_size=1000)
).add_stage(
    HtmlExtractorStage(preserve_structure=True)
).add_stage(
    DocumentFilterStage([
        length_filter(min_length=100)
    ])
)

# Build and execute
plan = pipeline.build()
executor = XennaExecutor(config={'batch_size': 100})
results = executor.execute(plan, initial_data)
```

## Creating Custom Stages

To create a custom stage, inherit from `ProcessingStage`:

```python
from ray_curator.stages import ProcessingStage, StageType
from ray_curator.data import Task
from typing import Optional

class MyCustomStage(ProcessingStage):
    @property
    def name(self) -> str:
        return "my_custom_stage"
    
    @property
    def stage_type(self) -> StageType:
        return StageType.TRANSFORMER
    
    def process(self, task: Task) -> Optional[Task]:
        # Get dataframe
        df = task.to_pandas()
        
        # Transform the dataframe
        df['processed'] = df['content'].apply(self.transform_text)
        
        # Return new task
        return Task(
            task_id=f"{task.task_id}_processed",
            dataset_name=task.dataset_name,
            data=df,
            metadata=task.metadata
        )
    
    def transform_text(self, text: str) -> str:
        # Your transformation logic
        return text.lower()
```

## Creating Custom Filters

Filters work on pandas Series (dataframe rows):

```python
from typing import Callable
import pandas as pd

def quality_filter(min_quality: float) -> Callable[[pd.Series], bool]:
    """Filter documents by quality score."""
    def filter_func(row: pd.Series) -> bool:
        # Calculate quality score from content
        quality = len(row['content']) / 100  # Simple example
        return quality >= min_quality
    return filter_func

# Use in pipeline
pipeline.add_stage(
    DocumentFilterStage([
        quality_filter(min_quality=0.8),
        length_filter(min_length=100)
    ])
)
```

## Stage Fusion

The pipeline planner can automatically fuse compatible stages for better performance. To enable fusion, stages should declare what they can fuse with:

```python
@property
def can_fuse_with(self) -> List[StageType]:
    return [StageType.FILTER]  # Can be fused with filter stages
```

## Resource Management

Stages can declare their resource requirements:

```python
@property
def requires_gpu(self) -> bool:
    return True

@property
def gpu_memory_gb(self) -> float:
    return 16.0  # Requires 16GB GPU memory

@property
def cpu_cores(self) -> float:
    return 4.0  # Requires 4 CPU cores
```

The executor will automatically allocate appropriate resources based on these declarations.

## Working with Dataframes

The `Task` class provides convenient methods for dataframe operations:

```python
# Convert between formats
df = task.to_pandas()  # Get as pandas DataFrame
table = task.to_pyarrow()  # Get as PyArrow Table

# Utility methods
task.filter_empty()  # Remove documents with empty content
task.sample(100)  # Sample 100 documents

# Access properties
num_docs = task.num_documents
``` 