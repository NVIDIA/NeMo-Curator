from nemo_curator.download.doc_builder import (
    DocumentExtractor,
)

from ray_curator.stages.download.text.utils import decode_html, lang_detect

from .html_extractors import HTMLExtractorAlgorithm, JusTextExtractor
from .utils import get_stop_list_dict


class CommonCrawlWARCExtractor(DocumentExtractor):
    def __init__(
        self,
        algorithm: HTMLExtractorAlgorithm | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
    ):
        if algorithm is None:
            algorithm = JusTextExtractor()

        if stop_lists is not None:
            self._stop_lists = stop_lists
        else:
            self._stop_lists = get_stop_list_dict()

        self.algorithm = algorithm
        super().__init__()

    def extract(self, content: str) -> dict[str, str] | None:
        html = decode_html(content)
        if html is not None:
            # Language detection and HTML extraction
            lang = lang_detect(html)
            text = None
            if lang in self._stop_lists:
                text = self.algorithm.extract_text(html, self._stop_lists[lang], lang)
            if text is not None:
                if len(text) > 0:
                    text = "\n\n".join(text)
                    return {"language": lang, "text": text}
                else:
                    return None
        return None
