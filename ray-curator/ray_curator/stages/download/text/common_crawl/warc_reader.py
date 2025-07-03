import os
from collections.abc import Iterator

from nemo_curator.download.doc_builder import (
    DocumentIterator,
)
from warcio.archiveiterator import ArchiveIterator


class CommonCrawlWARCIterator(DocumentIterator):
    def __init__(self, log_frequency: int = 1000):
        super().__init__()
        self._counter = 0
        self._log_frequency = log_frequency

    def iterate(self, file_path: str) -> Iterator[tuple[dict[str, str], str]]:
        # Loop over all records in the current WARC
        self._counter = 0
        bname = os.path.split(file_path)[-1]
        with open(file_path, "rb") as file_pointer:
            ai = ArchiveIterator(file_pointer, arc2warc=True)
            for _k, rec in enumerate(ai):
                # Get the response from the crawl
                if rec.rec_type == "response":
                    if self._counter > 0 and self._counter % self._log_frequency == 0:
                        print(f"Extracted {self._counter} records in WARC")
                    self._counter += 1
                    content = rec.content_stream().read()
                    warc_id = rec.rec_headers.get_header("WARC-Record-ID")[10:-1]
                    url = rec.rec_headers.get_header("WARC-Target-URI")
                    meta = {
                        "url": url,
                        "warc_id": warc_id,
                        "source_id": f"{bname}",
                    }
                    yield meta, content
