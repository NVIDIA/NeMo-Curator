from dataclasses import dataclass

import pandas as pd
from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.download.text.utils import decode_html, lang_detect
from ray_curator.tasks import DocumentBatch

from .html_extractors import HTMLExtractorAlgorithm
from .html_extractors.justext import JusTextExtractor
from .html_extractors.resiliparse import ResiliparseExtractor
from .html_extractors.trafilatura import TrafilaturaExtractor
from .utils import get_stop_list_dict


@dataclass
class CommonCrawlHTMLExtractor(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(
        self,
        algorithm: HTMLExtractorAlgorithm | str | None = None,
        algorithm_kwargs: dict | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
    ):
        super().__init__()
        algorithm_kwargs = algorithm_kwargs or {}
        if algorithm is None:
            logger.warning("No algorithm provided, using justext with default parameters")
            algorithm = JusTextExtractor()
        elif isinstance(algorithm, str):
            if algorithm == "justext":
                algorithm = JusTextExtractor(**algorithm_kwargs)
            elif algorithm == "resiliparse":
                algorithm = ResiliparseExtractor(**algorithm_kwargs)
            elif algorithm == "trafilatura":
                algorithm = TrafilaturaExtractor(**algorithm_kwargs)
        elif isinstance(algorithm, HTMLExtractorAlgorithm):
            if algorithm_kwargs:
                logger.warning("Algorithm kwargs provided are ignored when an HTMLExtractorAlgorithm is provided")
        else:
            msg = f"Invalid algorithm: {algorithm}"
            raise ValueError(msg)

        if stop_lists is not None:
            self._stop_lists = stop_lists
        else:
            self._stop_lists = get_stop_list_dict()

        self.algorithm = algorithm
        self.time_taken_decode_html = 0
        self.time_taken_lang_detect = 0
        self.time_taken_extract_text = 0

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process WARC content and extract text into a DocumentBatch."""
        records = []
        for _, row in task.to_pandas().iterrows():
            extracted = self.extract(row["content"])  # TODO: investigate exract needs bytes or str
            if extracted:
                records.append(
                    {
                        "url": row["url"],
                        "warc_id": row["warc_id"],
                        "source_id": row["source_id"],
                        "language": extracted["language"],
                        "text": extracted["text"],
                    }
                )

        table = pd.DataFrame(records)
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=table,
            _stage_perf=task._stage_perf,
        )

    def extract(self, content: bytes) -> dict[str, str] | None:
        html = decode_html(content)

        if html is not None:
            # Language detection and HTML extraction
            lang = lang_detect(html)

            text = None
            # TODO: Understand more on why we need to check for stop_lists here and why only
            # few of the records make it
            if lang in self._stop_lists:
                text = self.algorithm.extract_text(html, self._stop_lists[lang], lang)

            if text is not None:
                if len(text) > 0:
                    text = "\n\n".join(text)
                    return {"language": lang, "text": text}
                else:
                    return None
        return None

    @property
    def name(self) -> str:
        return "common_crawl_html_extractor"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "warc_id", "source_id", "content"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "warc_id", "source_id", "language", "text"]
