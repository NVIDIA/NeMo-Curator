import pandas as pd
import pytest

from ray_curator.stages.download.text.common_crawl.html_extractor import CommonCrawlHTMLExtractor
from ray_curator.stages.download.text.common_crawl.html_extractors import JusTextExtractor, ResiliparseExtractor
from ray_curator.tasks import DocumentBatch


class TestCommonCrawlHTMLExtractor:
    """Test suite for CommonCrawlHTMLExtractor."""

    def test_html_extractor_justext(self) -> None:
        """Test HTML extractor with JusText algorithm."""
        extractor = CommonCrawlHTMLExtractor(algorithm=JusTextExtractor())
        html = (
            "<html><body><p>Common Crawl test paragraph for justext extractor. "
            "Four score and seven years ago our fathers brought forth on this continent a new nation, "
            "conceived in liberty, and dedicated to the proposition that all men are created equal.</p></body></html>"
        )
        content = html.encode("utf-8")
        result = extractor.extract(content)

        assert result is not None
        # The extracted text should include our test paragraph.
        assert "Common Crawl test paragraph for justext extractor." in result["text"]
        assert "language" in result

    def test_html_extractor_resiliparse(self) -> None:
        """Test HTML extractor with Resiliparse algorithm."""
        extractor = CommonCrawlHTMLExtractor(algorithm=ResiliparseExtractor())
        html = (
            "<html><body><p>Common Crawl test paragraph for resiliparse extractor. "
            "Four score and seven years ago our fathers brought forth on this continent a new nation, "
            "conceived in liberty, and dedicated to the proposition that all men are created equal.</p></body></html>"
        )
        content = html.encode("utf-8")
        result = extractor.extract(content)

        assert result is not None
        assert "Common Crawl test paragraph for resiliparse extractor." in result["text"]
        assert "language" in result

    def test_html_extractor_process_method(self) -> None:
        """Test the process method with DocumentBatch."""
        # Create test data with English content that should pass language detection and stopword filtering
        data = pd.DataFrame(
            [
                {
                    "url": "http://example1.com",
                    "warc_id": "warc1",
                    "source_id": "source1",
                    "content": b"<html><body><p>This is a comprehensive test paragraph with many important words and meaningful content. "
                    b"We are testing the HTML extraction functionality to ensure that it works properly and efficiently "
                    b"with various types of content. The system should be able to process this text correctly.</p></body></html>",
                },
                {
                    "url": "http://example2.com",
                    "warc_id": "warc2",
                    "source_id": "source2",
                    "content": b"<html><body><p>Another comprehensive test paragraph with sufficient and meaningful content for testing purposes. "
                    b"This content contains various words and phrases that should be properly extracted by the HTML processing system. "
                    b"The extraction algorithm should successfully identify and process this English text content.</p></body></html>",
                },
            ]
        )

        task = DocumentBatch(task_id="test_task", dataset_name="test_dataset", data=data, _stage_perf=[], _metadata={})

        extractor = CommonCrawlHTMLExtractor(algorithm="justext")
        result = extractor.process(task)

        # Verify result structure
        assert isinstance(result, DocumentBatch)
        assert result.task_id == task.task_id
        assert result.dataset_name == task.dataset_name

        # Check output data
        output_df = result.to_pandas()
        assert len(output_df) >= 1  # At least some records should pass extraction

        # Check required columns
        expected_columns = ["url", "warc_id", "source_id", "language", "text"]
        for col in expected_columns:
            assert col in output_df.columns

    def test_html_extractor_properties(self) -> None:
        """Test HTML extractor properties and methods."""
        extractor = CommonCrawlHTMLExtractor(algorithm="justext")

        assert extractor.name == "common_crawl_html_extractor"

        inputs = extractor.inputs()
        assert inputs == (["data"], ["url", "warc_id", "source_id", "content"])

        outputs = extractor.outputs()
        assert outputs == (["data"], ["url", "warc_id", "source_id", "language", "text"])

    def test_html_extractor_invalid_algorithm(self) -> None:
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Invalid algorithm"):
            # Test with an invalid type (ignoring type checking for test purposes)
            CommonCrawlHTMLExtractor(algorithm=123)  # type: ignore[reportArgumentType]

    def test_html_extractor_string_algorithms(self) -> None:
        """Test that string algorithm names work correctly."""
        # Test justext
        extractor_justext = CommonCrawlHTMLExtractor(algorithm="justext")
        assert isinstance(extractor_justext.algorithm, JusTextExtractor)

        # Test resiliparse
        extractor_resiliparse = CommonCrawlHTMLExtractor(algorithm="resiliparse")
        assert isinstance(extractor_resiliparse.algorithm, ResiliparseExtractor)

    def test_html_extractor_no_algorithm(self) -> None:
        """Test that default algorithm (justext) is used when none specified."""
        extractor = CommonCrawlHTMLExtractor()
        assert isinstance(extractor.algorithm, JusTextExtractor)
