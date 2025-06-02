"""Example demonstrating composite stages that decompose into execution stages."""

import re
from typing import List, Optional

import pandas as pd

from ray_curator.data import Task, DocumentBatch
from ray_curator.stages.base import CompositeStage, ProcessingStage, StageType


class TextCleaningStage(ProcessingStage):
    """Low-level stage that cleans text content."""
    
    def __init__(self, name: str = "text_cleaning"):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def stage_type(self) -> StageType:
        return StageType.TRANSFORMER
    
    def process(self, task: Task) -> Optional[Task]:
        """Clean text content by normalizing whitespace and removing unwanted characters."""
        if not hasattr(task, 'to_pandas'):
            return task
            
        df = task.to_pandas()
        
        if 'content' not in df.columns:
            return task
        
        # Clean text: normalize whitespace, remove control characters
        df['content'] = df['content'].apply(self._clean_text)
        
        # Create new task with cleaned data
        return DocumentBatch(
            task_id=f"{task.task_id}_cleaned",
            dataset_name=task.dataset_name,
            data=df,
            metadata={**task.metadata, "stage": self.name}
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text content."""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()


class TextFilterStage(ProcessingStage):
    """Low-level stage that filters out low-quality text."""
    
    def __init__(self, min_length: int = 100, max_length: int = 50000, name: str = "text_filter"):
        self.min_length = min_length
        self.max_length = max_length
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def stage_type(self) -> StageType:
        return StageType.FILTER
    
    def process(self, task: Task) -> Optional[Task]:
        """Filter text based on length and quality criteria."""
        if not hasattr(task, 'to_pandas'):
            return task
            
        df = task.to_pandas()
        
        if 'content' not in df.columns:
            return task
        
        # Calculate text lengths
        df['text_length'] = df['content'].str.len()
        
        # Filter by length
        mask = (df['text_length'] >= self.min_length) & (df['text_length'] <= self.max_length)
        
        # Additional quality filters
        mask &= ~df['content'].str.contains(r'^[\s\W]*$', regex=True, na=False)  # Not just whitespace/punctuation
        mask &= df['content'].str.count(r'[a-zA-Z]') >= 50  # At least 50 letters
        
        filtered_df = df[mask].drop('text_length', axis=1)
        
        if len(filtered_df) == 0:
            return None
        
        return DocumentBatch(
            task_id=f"{task.task_id}_filtered",
            dataset_name=task.dataset_name,
            data=filtered_df,
            metadata={
                **task.metadata, 
                "stage": self.name,
                "original_count": len(df),
                "filtered_count": len(filtered_df)
            }
        )


class LanguageDetectionStage(ProcessingStage):
    """Low-level stage that adds language detection."""
    
    def __init__(self, name: str = "language_detection"):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def stage_type(self) -> StageType:
        return StageType.TRANSFORMER
    
    @property
    def requires_gpu(self) -> bool:
        return False  # Could be True if using GPU-based language detection
    
    def process(self, task: Task) -> Optional[Task]:
        """Add language detection to documents."""
        if not hasattr(task, 'to_pandas'):
            return task
            
        df = task.to_pandas()
        
        if 'content' not in df.columns:
            return task
        
        # Simple language detection (in practice, would use a real library)
        df['language'] = df['content'].apply(self._detect_language)
        
        return DocumentBatch(
            task_id=f"{task.task_id}_with_language",
            dataset_name=task.dataset_name,
            data=df,
            metadata={**task.metadata, "stage": self.name}
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (placeholder implementation)."""
        if not isinstance(text, str) or len(text) < 10:
            return "unknown"
        
        # Very simple heuristic - in practice use proper language detection
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)
        
        return "en" if english_count >= 3 else "other"


class TextPreprocessingPipeline(CompositeStage):
    """High-level composite stage for comprehensive text preprocessing.
    
    This user-facing stage provides a simple API for text preprocessing
    but decomposes into multiple specialized execution stages.
    """
    
    def __init__(
        self, 
        min_length: int = 100,
        max_length: int = 50000,
        include_language_detection: bool = True,
        name: str = "text_preprocessing_pipeline"
    ):
        """Initialize the text preprocessing pipeline.
        
        Args:
            min_length: Minimum text length for filtering
            max_length: Maximum text length for filtering  
            include_language_detection: Whether to include language detection
            name: Name for this composite stage
        """
        self.min_length = min_length
        self.max_length = max_length
        self.include_language_detection = include_language_detection
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def decompose(self) -> List[ProcessingStage]:
        """Decompose into execution stages."""
        stages = []
        
        # Always include cleaning and filtering
        stages.append(TextCleaningStage(name=f"{self.name}_cleaning"))
        stages.append(TextFilterStage(
            min_length=self.min_length,
            max_length=self.max_length,
            name=f"{self.name}_filtering"
        ))
        
        # Optionally include language detection
        if self.include_language_detection:
            stages.append(LanguageDetectionStage(name=f"{self.name}_language"))
        
        return stages
    
    def get_description(self) -> str:
        """Get description of this composite stage."""
        desc = (
            f"Text preprocessing pipeline that performs cleaning, "
            f"filtering (length: {self.min_length}-{self.max_length})"
        )
        if self.include_language_detection:
            desc += ", and language detection"
        return desc


# Example usage
def create_example_pipeline():
    """Create an example pipeline using the composite stage."""
    from ray_curator.pipeline.pipeline import Pipeline
    
    # Create pipeline with high-level composite stage
    pipeline = Pipeline(
        name="text_processing_example",
        description="Example using composite stages"
    )
    
    # Add the high-level preprocessing stage
    # Users only need to configure high-level parameters
    preprocessing = TextPreprocessingPipeline(
        min_length=50,
        max_length=10000,
        include_language_detection=True
    )
    
    pipeline.add_stage(preprocessing)
    
    return pipeline


def demonstrate_decomposition():
    """Demonstrate how the composite stage decomposes."""
    
    # Create the composite stage
    composite = TextPreprocessingPipeline(
        min_length=100,
        max_length=5000,
        include_language_detection=True
    )
    
    print(f"Composite Stage: {composite.name}")
    print(f"Description: {composite.get_description()}")
    print(f"Is Composite: {composite.is_composite()}")
    print()
    
    # Show decomposition
    execution_stages = composite.decompose()
    print(f"Decomposes into {len(execution_stages)} execution stages:")
    
    for i, stage in enumerate(execution_stages, 1):
        print(f"  {i}. {stage.name} ({stage.stage_type.value})")
        if stage.requires_gpu:
            print(f"     - GPU Required: {stage.gpu_memory_gb}GB")
        print(f"     - CPU Cores: {stage.cpu_cores}")
    
    return execution_stages


if __name__ == "__main__":
    # Demonstrate the decomposition
    stages = demonstrate_decomposition()
    
    print("\nExample pipeline creation:")
    pipeline = create_example_pipeline()
    print(pipeline.describe()) 