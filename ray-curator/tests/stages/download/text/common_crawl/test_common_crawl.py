# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import pytest
from nemo_curator.download import (
    JusTextExtractor,
    ResiliparseExtractor,
    TrafilaturaExtractor,
    download_and_extract,
)
from nemo_curator.download.commoncrawl import (
    CommonCrawlWARCDownloader,
    CommonCrawlWARCExtractor,
    CommonCrawlWARCIterator,
    get_common_crawl_urls,
    get_stop_list_dict,
)
from pytest import MonkeyPatch


class FakeCompletedProcess:
    def __init__(self) -> None:
        self.returncode = 0


def fake_run_success(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
    return FakeCompletedProcess()


@pytest.fixture
def html_string() -> str:
    # Modified from https://github.com/chatnoir-eu/chatnoir-resiliparse/blob/abdf1966fb3cefe3e0790e510ab5cb1446f99a79/tests/resiliparse/extract/test_html2text.py
    return """<!doctype html>
        <head>
            <title>My Title</title>
            <meta charset="utf-8">
            <style>* { margin: 0; }</style>
        </head>
        <body>
            <section id="wrapper">
                <nav>
                    <ul>
                        <li>Nav 1</li>
                        <li>
                            <p>Nav 2</p>
                            <ul>
                                <li><p>Nav 3</p></li>
                            </ul>
                        </li>
                    </ul>
                </nav>
                <main>
                    This is a sample paragraph. In it we write words.
                    These are stopwords: because did than has near we almost while what still.
                    <a href="#foo" hidden>bar</a>

                    <p>
                    This paragraph doesn't have many stopwords. Remove it.
                    <br>Let's keep this paragraph: either came does last new took taken making became from.
                    </p>

                    <button aria-hidden="true">Click here</button>
                    <input type="hidden" value="foo">
                    <input type="text" value="Some text" placeholder="Insert text">
                    <input type="text" placeholder="Insert text">
                    <img src="" alt="Some image">
                    <object data="" class="some-class hidden">Cannot display object</object>
                </main>
                <script language="vbscript" type="text/vbscript">MsgBox("Hello World!")</script>
                <noscript>Sorry, your browser doesn't support VB Script!</noscript>
                <div><div><div><footer id="global-footer">
                    Copyright (C) 2021 Foo Bar
                </footer></div></div></div>
            </section>
        </body>
    </html>"""


class TestDownload:
    def test_imports(self) -> None:
        from nemo_curator.download import (
            JusTextExtractor,  # noqa: F401
            ResiliparseExtractor,  # noqa: F401
            TrafilaturaExtractor,  # noqa: F401
            download_arxiv,  # noqa: F401
            download_common_crawl,  # noqa: F401
            download_wikipedia,  # noqa: F401
        )

    @pytest.mark.skip(reason="This test is flaky due to calling out to an external service and should be fixed.")
    def test_incorrect_snapshot_order(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011, PT012
            end_snapshot = "2021-04"
            start_snapshot = "2021-10"
            _urls = get_common_crawl_urls(start_snapshot, end_snapshot)

    @pytest.mark.skip(reason="This test is flaky due to calling out to an external service and should be fixed.")
    def test_incorrect_snapshot_order_news(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011, PT012
            end_snapshot = "2021-04"
            start_snapshot = "2021-10"
            _urls = get_common_crawl_urls(start_snapshot, end_snapshot, news=True)

    def test_no_urls(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011, PT012
            output_format = {
                "text": str,
            }
            download_and_extract(
                [],
                [],
                CommonCrawlWARCDownloader(download_dir="."),
                CommonCrawlWARCIterator(),
                CommonCrawlWARCExtractor(),
                output_format,
            )

    def test_url_path_mismatch(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011, PT012
            output_format = {
                "text": str,
            }
            download_and_extract(
                ["one", "two", "three"],
                ["one"],
                CommonCrawlWARCDownloader(download_dir="."),
                CommonCrawlWARCIterator(),
                CommonCrawlWARCExtractor(),
                output_format,
            )


class TestCommonCrawl:
    def test_common_crawl_downloader_existing_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        # Create a temporary downloads directory and simulate an already-downloaded file.
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")  # "commoncrawl.warc"
        file_path = os.path.join(str(download_dir), output_name)
        # Write dummy content to simulate an existing download.
        with open(file_path, "w") as f:
            f.write("existing content")

        downloader = CommonCrawlWARCDownloader(str(download_dir), aws=False, verbose=False)

        # Monkey-patch subprocess.run to track if it gets called.
        called_run = False

        def fake_run(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
            nonlocal called_run
            called_run = True
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = downloader.download(url)
        assert result == file_path
        # Since the file already exists, no download should be attempted.
        assert not called_run

    def test_common_crawl_downloader_new_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        # Create a temporary downloads directory; ensure the file does not exist.
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        url = "http://dummy/commoncrawl.warc"
        parsed = urlparse(url)
        output_name = parsed.path[1:].replace("/", "-")  # "commoncrawl.warc"
        file_path = os.path.join(str(download_dir), output_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        downloader = CommonCrawlWARCDownloader(str(download_dir), aws=False, verbose=False)

        called_run = False

        def fake_run(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
            nonlocal called_run
            called_run = True
            return FakeCompletedProcess()

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = downloader.download(url)
        assert result == file_path
        # Since the file did not exist, a download call (and subprocess.run) should have been made.
        assert called_run

    def test_common_crawl_iterator(self, tmp_path: Path) -> None:
        # Create a minimal valid WARC file with a single "response" record.
        raw_warc_path = tmp_path / "dummy.warc"
        http_response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "\r\n"
            "<html><body><p>Common Crawl test paragraph with some content.</p></body></html>\r\n"
        )
        http_response_bytes = http_response.encode("utf-8")
        content_length = len(http_response_bytes)
        warc_record = (
            (
                f"WARC/1.0\r\n"
                f"WARC-Type: response\r\n"
                f"WARC-Record-ID: <urn:uuid:1234>\r\n"
                f"WARC-Date: 2022-01-01T00:00:00Z\r\n"
                f"WARC-Target-URI: http://example.com\r\n"
                f"Content-Length: {content_length}\r\n"
                f"\r\n"
            ).encode()
            + http_response_bytes
            + b"\r\n\r\n"
        )
        raw_warc_path.write_bytes(warc_record)

        iterator = CommonCrawlWARCIterator(log_frequency=1)
        records = list(iterator.iterate(str(raw_warc_path)))
        assert len(records) == 1
        meta, content = records[0]
        # Check that the URL from the header is captured.
        assert "example.com" in meta["url"]
        # Verify that the content includes our test paragraph.
        assert b"Common Crawl test paragraph" in content

    def test_common_crawl_extractor_justext(self) -> None:
        extractor = CommonCrawlWARCExtractor(algorithm=JusTextExtractor())
        html = (
            "<html><body><p>Common Crawl test paragraph for justext extractor. "
            "Four score and seven years ago our fathers brought forth on this continent a new nation, "
            "conceived in liberty, and dedicated to the proposition that all men are created equal.</p></body></html>"
        )
        content = html.encode("utf-8")
        result = extractor.extract(content)
        print(result)
        assert result is not None
        # The extracted text should include our test paragraph.
        assert "Common Crawl test paragraph for justext extractor." in result["text"]
        assert "language" in result

    def test_common_crawl_extractor_resiliparse(self) -> None:
        extractor = CommonCrawlWARCExtractor(algorithm=ResiliparseExtractor())
        html = (
            "<html><body><p>Common Crawl test paragraph for resiliparse extractor. "
            "Four score and seven years ago our fathers brought forth on this continent a new nation, "
            "conceived in liberty, and dedicated to the proposition that all men are created equal.</p></body></html>"
        )
        content = html.encode("utf-8")
        result = extractor.extract(content)
        print(result)
        assert result is not None
        assert "Common Crawl test paragraph for resiliparse extractor." in result["text"]
        assert "language" in result


class TestExtractor:
    def test_resiliparse_extract_text(self, html_string: str) -> None:
        algorithm = ResiliparseExtractor()
        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(html_string, stop_words["ENGLISH"], "ENGLISH")

        expected = [
            "This is a sample paragraph. In it we write words. These are stopwords: because did than has near we almost while what still.",
            "Let's keep this paragraph: either came does last new took taken making became from.",
        ]

        assert result == expected

    def test_trafilatura_extract_text(self, html_string: str) -> None:
        algorithm = TrafilaturaExtractor(
            min_extracted_size=10,
            min_duplcheck_size=10,
            max_repetitions=1,
            deduplicate=True,
        )
        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(html_string, stop_words["ENGLISH"], "ENGLISH")

        expected = [
            "Let's keep this paragraph: either came does last new took taken making became from.",
        ]

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_thai_text(self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]) -> None:
        thai_html = """<!doctype html>
            <head>
                <title>ชื่อเรื่องของฉัน</title>
            </head>
            <body>
                    <main>
                        นี่คือตัวอย่างย่อหน้า ในนั้นเราเขียนคำต่างๆ
                        เหล่านี้เป็นคำหยุด: เพราะว่า ทำ กว่า มี ใกล้ เรา เกือบจะ ขณะที่ อะไร ยังคง

                        <p>
                        ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก
                        <br>เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก
                        </p>

                    </main>
            </body>
        </html>"""

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "นี่คือตัวอย่างย่อหน้า ในนั้นเราเขียนคำต่างๆ\nเหล่านี้เป็นคำหยุด: เพราะว่า ทำ กว่า มี ใกล้ เรา เกือบจะ ขณะที่ อะไร ยังคง",
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก\nเรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "นี่คือตัวอย่างย่อหน้า ในนั้นเราเขียนคำต่างๆ เหล่านี้เป็นคำหยุด: เพราะว่า ทำ กว่า มี ใกล้ เรา เกือบจะ ขณะที่ อะไร ยังคง",
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก",
                "เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก",
                "เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก",
                "เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(thai_html, stop_words["THAI"], "THAI")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_chinese_text(
        self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        chinese_html = """<!doctype html>
            <head>
                <title>我的标题</title>
            </head>
            <body>
                    <main>
                        这是一个示例段落。我们在其中写下单词。

                        <p>
                        本段落没有太多停用词。请将其删除。
                        <br>让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。
                        </p>

                    </main>
            </body>
        </html>"""  # noqa: RUF001

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "这是一个示例段落。我们在其中写下单词。",
                "本段落没有太多停用词。请将其删除。\n让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。",  # noqa: RUF001
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "这是一个示例段落。我们在其中写下单词。",
                "本段落没有太多停用词。请将其删除。",
                "让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。",  # noqa: RUF001
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "这是一个示例段落。我们在其中写下单词。",
                "本段落没有太多停用词。请将其删除。",
                "让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。",  # noqa: RUF001
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(chinese_html, stop_words["CHINESE"], "CHINESE")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_japanese_text(
        self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        japanese_html = """<!doctype html>
            <head>
                <title>私のタイトル</title>
            </head>
            <body>
                    <main>
                        これはサンプルの段落です。ここに単語を書き込みます。

                        <p>
                        この段落にはストップワードがあまりありません。削除してください。
                        <br>この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。
                        </p>

                    </main>
            </body>
        </html>"""

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "これはサンプルの段落です。ここに単語を書き込みます。",
                "この段落にはストップワードがあまりありません。削除してください。\nこの段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "これはサンプルの段落です。ここに単語を書き込みます。",
                "この段落にはストップワードがあまりありません。削除してください。",
                "この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "この段落にはストップワードがあまりありません。削除してください。",
                "この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
                "この段落にはストップワードがあまりありません。削除してください。",
                "この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(japanese_html, stop_words["JAPANESE"], "JAPANESE")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_korean_text(self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]) -> None:
        korean_html = """<!doctype html>
            <head>
                <title>내 제목</title>
            </head>
            <body>
                    <main>
                        이것은 샘플 문단입니다. 여기에 단어를 적습니다.
                        이것들은 불용어입니다: 왜냐하면, 했으므로, 보다, 가까이에, 우리, 거의, 동안, 무엇, 아직도.

                        <p>
                        이 문단에는 불용어가 많지 않습니다. 제거하세요.
                        <br>이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.
                        </p>

                    </main>
            </body>
        </html>"""

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "이것은 샘플 문단입니다. 여기에 단어를 적습니다.\n이것들은 불용어입니다: 왜냐하면, 했으므로, 보다, 가까이에, 우리, 거의, 동안, 무엇, 아직도.",
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.\n이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "이것은 샘플 문단입니다. 여기에 단어를 적습니다. 이것들은 불용어입니다: 왜냐하면, 했으므로, 보다, 가까이에, 우리, 거의, 동안, 무엇, 아직도.",
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.",
                "이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.",
                "이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.",
                "이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(korean_html, stop_words["KOREAN"], "KOREAN")

        assert result == expected
