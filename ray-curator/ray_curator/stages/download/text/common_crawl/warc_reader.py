from pathlib import Path

import pandas as pd
from loguru import logger
from warcio.archiveiterator import ArchiveIterator  # TODO: consider using fastwarc

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch, FileGroupTask


class WarcReader(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Processes downloaded WARC files."""

    def __init__(self):
        """
        Initialize the WARC processor.

        Args:
            log_frequency: How often to log progress
        """
        super().__init__()

    def process(self, task: FileGroupTask) -> DocumentBatch:
        """Process a task containing WARC files and extract their contents."""
        all_results: list[dict] = []
        for warc_path in task.data:
            results = self.process_warc(warc_path)
            all_results.extend(results)

        data_df = pd.DataFrame(all_results)

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=data_df,
            _stage_perf=task._stage_perf,
        )

    def process_warc(self, warc_path: Path | str) -> list[dict]:
        """
        Process a WARC file and extract its contents.

        Args:
            warc_path: Path to the WARC file

        Returns:
            List of dictionaries containing extracted content and metadata
        """
        results = []
        # This should support cloud paths so that we can directly process WARC files on S3
        filename = warc_path.name if isinstance(warc_path, Path) else warc_path.split("/")[-1]
        num_records = 0
        with open(warc_path, "rb") as file_pointer:
            archive_iterator = ArchiveIterator(file_pointer, arc2warc=True)
            while True:
                try:
                    rec = next(archive_iterator)
                    if rec.rec_type == "response":
                        content = rec.content_stream().read()
                        warc_id = rec.rec_headers.get_header("WARC-Record-ID")[10:-1]
                        url = rec.rec_headers.get_header("WARC-Target-URI")
                        results.append({"url": url, "warc_id": warc_id, "source_id": filename, "content": content})
                        num_records += 1
                except StopIteration:  # noqa: PERF203
                    # End of file reached normally
                    break
                except Exception as e:  # noqa: BLE001
                    # Handle corruption or other errors
                    logger.error(f"Error processing record {num_records} in {filename}: {e!s}")
                    # Try to continue with next record
                    continue
        return results

    @property
    def name(self) -> str:
        return "warc_processor"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "warc_id", "source_id", "content"]
