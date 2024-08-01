

import os
from typing import Set, Tuple

import requests

from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
)

class TedTalksDownloader(DocumentDownloader): 
    def __init__(self, download_dir: str):
        super().__init__()

        if not os.path.isdir(download_dir):
            os.makedirs(download_dir)

        self._download_dir = download_dir
        print("Download directory: ", self._download_dir)

    def download(self, url_src: str, url_tgt: str, force=False) -> str:
        src_filename = os.path.basename(url_src)
        src_out = os.path.join(self._download_dir, src_filename)
        tgt_filename = os.path.basename(url_tgt)
        tgt_out = os.path.join(self._download_dir, tgt_filename)

        
        sides = ['src', 'tgt']
        urls = [url_src, url_tgt]
        output_files = {'src': src_out, 'tgt': tgt_out}
        for side, url, out_file_key in zip(sides, urls, output_files):
            if os.path.exists(output_files[out_file_key]) and force is False:
                print(f"File '{output_files[out_file_key]}' already exists, skipping download.")
            else:
                print(f"Downloading TED Talks dataset from '{url}'...")
                response = requests.get(url)

                with open(output_files[out_file_key], "wb") as file:
                    file.write(response.content)

        return output_files


class TedTalksIterator(DocumentIterator):
    def __init__(self):
        super().__init__()
        self._counter = -1
    def iterate(self, file_path):
        self._counter = -1
        file_name = os.path.basename(file_path)

        with open(file_path, "r") as file:
            example = []
            def split_meta(example):
                if example:
                    self._counter += 1
                    content = example
                    meta = {
                        "filename": file_name,
                        "id": f"{file_name}-{self._counter}",
                    }

                    return meta, content
                return
            for line in file:
                example = line.strip()
                if example:
                    yield split_meta(example)
                    example = []
            if example:
                yield split_meta(example)

class TedTalksExtractor(DocumentExtractor):
    def extract(self, content: str) -> Tuple[Set, str]:
        # No metadata for the text, just the content.
        return {}, content
