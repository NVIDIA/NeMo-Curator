from ray_curator.modifiers.base import DocumentModifier
from ray_curator.stages.base import ProcessingStage
from ray_curator.data.task import DocumentBatch


class ModifierStage(ProcessingStage[DocumentBatch]):
    def __init__(self, modifier: DocumentModifier, modify_column: str = "text"):
        super().__init__()
        self.modifier = modifier
        self.modify_column = modify_column

    @property
    def name(self) -> str:
        return f"modify[{self.modifier.name}]"

    def setup(self) -> None:
        self.modifier.setup()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """
        Modifies the text in the dataset
        Args:
            batch (DocumentBatch): The batch to apply the modifier to
        Returns:
            DocumentBatch: A batch with the modifier applied
        """
        df = batch.to_pandas()
        df[self.modify_column] = df[self.modify_column].apply(self.modifier.modify_document)

        # Create output batch
        output_batch = DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            metadata={**batch.metadata, "modify": self.name},
            text_column=batch.text_column,
            id_column=batch.id_column,
            additional_columns=batch.additional_columns,
        )
        output_batch.add_stage(self.name)

        return output_batch