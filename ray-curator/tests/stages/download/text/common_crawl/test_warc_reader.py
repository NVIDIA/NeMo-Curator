from pathlib import Path

from ray_curator.stages.download.text.common_crawl.warc_reader import WarcReader
from ray_curator.tasks import DocumentBatch, FileGroupTask


class TestWarcReader:
    """Test suite for WarcReader."""

    def test_warc_reader_basic(self, tmp_path: Path) -> None:
        """Test that WarcReader can process a minimal WARC file."""
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

        # Create FileGroupTask with the WARC file
        task = FileGroupTask(
            task_id="test_task", dataset_name="test_dataset", data=[str(raw_warc_path)], _stage_perf=[], _metadata={}
        )

        reader = WarcReader()
        result = reader.process(task)

        # Verify the result
        assert isinstance(result, DocumentBatch)
        assert result.task_id == task.task_id
        assert result.dataset_name == task.dataset_name

        # Convert to pandas to check content
        df = result.to_pandas()
        assert len(df) == 1

        # Check that the URL from the header is captured
        assert "example.com" in df.iloc[0]["url"]
        # Verify that the content includes our test paragraph
        assert b"Common Crawl test paragraph" in df.iloc[0]["content"]
        # Check other columns
        assert "warc_id" in df.columns
        assert "source_id" in df.columns
        assert df.iloc[0]["warc_id"] == "1234"

    def test_warc_reader_properties(self) -> None:
        """Test WarcReader properties and methods."""
        reader = WarcReader()

        assert reader.name == "warc_processor"

        inputs = reader.inputs()
        assert inputs == (["data"], [])

        outputs = reader.outputs()
        assert outputs == (["data"], ["url", "warc_id", "source_id", "content"])

    def test_warc_reader_multiple_files(self, tmp_path: Path) -> None:
        """Test WarcReader with multiple WARC files."""
        # Create two WARC files
        files = []
        for i in range(2):
            raw_warc_path = tmp_path / f"dummy_{i}.warc"
            http_response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/html\r\n"
                "\r\n"
                f"<html><body><p>Test content {i}</p></body></html>\r\n"
            )
            http_response_bytes = http_response.encode("utf-8")
            content_length = len(http_response_bytes)
            warc_record = (
                (
                    f"WARC/1.0\r\n"
                    f"WARC-Type: response\r\n"
                    f"WARC-Record-ID: <urn:uuid:123{i}>\r\n"
                    f"WARC-Date: 2022-01-01T00:00:00Z\r\n"
                    f"WARC-Target-URI: http://example{i}.com\r\n"
                    f"Content-Length: {content_length}\r\n"
                    f"\r\n"
                ).encode()
                + http_response_bytes
                + b"\r\n\r\n"
            )
            raw_warc_path.write_bytes(warc_record)
            files.append(str(raw_warc_path))

        # Create FileGroupTask with both WARC files
        task = FileGroupTask(
            task_id="test_task", dataset_name="test_dataset", data=files, _stage_perf=[], _metadata={}
        )

        reader = WarcReader()
        result = reader.process(task)

        # Verify the result
        df = result.to_pandas()
        assert len(df) == 2  # Should have records from both files

        # Check that both URLs are present
        urls = df["url"].tolist()
        assert "http://example0.com" in urls
        assert "http://example1.com" in urls
