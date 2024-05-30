# +
import json
import re

import dask.dataframe as dd
import pandas as pd

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    DocumentFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsFilter,
    RepeatingTopNGramsFilter,
    UrlsFilter,
    WordCountFilter,
)
from nemo_curator.filters.code import (
    NumberOfLinesOfCodeFilter,
    PythonCommentToCodeFilter,
)
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.modules import ExactDuplicates
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import add_distributed_args

from .modifiers import QuotationUnifier

# -


def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    """
    Cleans and unifies the given dataset using a set of predefined cleaners.

    Args:
        dataset (DocumentDataset): The dataset to be cleaned and unified.

    Returns:
        DocumentDataset: The cleaned and unified dataset.
    """
    cleaners = Sequential(
        [
            # Unify all the quotation marks
            Modify(QuotationUnifier(), text_field="text"),
            # Unify all unicode
            Modify(UnicodeReformatter(), text_field="text"),
        ]
    )
    return cleaners(dataset)


def filter_dataset(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given dataset based on various criteria.
    Refer to the full list of all filters here:
    https://github.com/NVIDIA/NeMo-Curator/blob/main/config/heuristic_filter_en.yaml
    https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/peft-curation/main.py

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            # If a document contains a number of words not
            # within a specified range then discard
            ScoreFilter(
                WordCountFilter(min_words=50, max_words=100000),
                text_field="text",
                score_field="word_count",
                score_type=int,
            ),
            # If the document shrinks by > x% in terms of number of characters after
            # removing the top n-grams then discard. Source: Gopher (Rae et al., 2021)
            ScoreFilter(
                RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
                text_field="text",
                score_type=float,
            ),
            # If the document shrinks by > 30% in terms of number of lines after
            # removing duplicate lines then discard. Source: Gopher (Rae et al., 2021)
            ScoreFilter(
                RepeatedLinesFilter(max_repeated_line_fraction=0.7),
                text_field="text",
                score_type=float,
            ),
            #  If the document shrinks by > 30% in terms of number of lines after
            # removing duplicate paragraphs then discard. Source: Gopher (Rae et al., 2021)
            ScoreFilter(
                RepeatedParagraphsFilter(max_repeated_paragraphs_ratio=0.7),
                text_field="text",
                score_type=float,
            ),
            # If more than 20% of the document is comprised of URLs then discard
            ScoreFilter(
                UrlsFilter(max_url_to_text_ratio=0.2),
                text_field="text",
                score_type=float,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def filter_code(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given dataset based on various criteria.
    Refer to the full list of all filters here:
    https://github.com/NVIDIA/NeMo-Curator/blob/main/config/heuristic_filter_en.yaml
    https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/peft-curation/main.py

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            # If a document contains a number of words not
            # within a specified range then discard
            ScoreFilter(
                WordCountFilter(min_words=5, max_words=100000),
                text_field="text",
                score_field="word_count",
                score_type=int,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def filter_code_dataset(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given code dataset based on various code-specific criteria.
    Refer to the full list of all filters here:
    https://github.com/NVIDIA/NeMo-Curator/blob/main/config/heuristic_filter_code.yaml
    https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/filters/code.py

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            # if the comment to code ratio is not within the specified range,
            # discard the document
            ScoreFilter(
                PythonCommentToCodeFilter(
                    min_comment_to_code_ratio=0.001, max_comment_to_code_ratio=0.80
                ),
                text_field="text",
                score_type=float,
            ),
            # if the number of lines of code is not within the specified range,
            # discard the document
            ScoreFilter(
                NumberOfLinesOfCodeFilter(min_lines=5, max_lines=20000),
                text_field="text",
                score_type=int,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


# +
def comment_redaction(json_data):
    data = json.loads(json_data)
    json_docs = data.get("documents", [])
    for doc in json_docs:
        if "text" in doc:
            # print("initial", doc["text"])

            # Extract the comment content between /* and */
            start_idx = doc["text"].find("/*")
            end_idx = doc["text"].find("*/")
            if start_idx != -1 and end_idx != -1:
                comment = doc["text"][start_idx : end_idx + 2]

                # Create a Dask DataFrame from the given data
                comment_df = dd.from_pandas(
                    pd.DataFrame([{"text": comment}]), npartitions=1
                )
                # Create a DocumentDataset instance
                comment_doc_dataset = DocumentDataset(comment_df)
                redacted_comment = redact_pii(comment_doc_dataset)

                # Extract the "text" column from the DocumentDataset
                text_column = redacted_comment.df["text"]

                # Convert the Dask Series to a Pandas Series (if needed)
                text_series = text_column.compute()

                # Get the first (and presumably only) value from the Series
                text_value = text_series.iloc[0]

                # Convert the value to a string
                redacted_text_string = str(text_value)

                doc["text"] = doc["text"].replace(comment, redacted_text_string)

                # print("final", doc["text"])

    # Serialize the updated dictionary back to a JSON string
    updated_json_data = json.dumps(data, indent=2)
    return updated_json_data


def redact(dataset: DocumentDataset) -> DocumentDataset:
    # Extract relevant data from the Dask DataFrame
    length = len(dataset)
    data_dict = {
        "documents": dataset.df.compute().to_dict(orient="records"),
    }
    # Serialize the dictionary to a JSON string
    json_data = json.dumps(data_dict, indent=2)
    redactd_json_data = comment_redaction(json_data)

    # Deserialize the JSON string to a dictionary
    redactd_data_dict = json.loads(redactd_json_data)

    # Create a DataFrame from the dictionary
    redactd_dataset_df = dd.from_pandas(
        pd.DataFrame(redactd_data_dict["documents"]), npartitions=length
    )

    # Create a DocumentDataset instance
    redactd_document_dataset = DocumentDataset(redactd_dataset_df)

    return redactd_document_dataset


def redact_pii(dataset: DocumentDataset) -> DocumentDataset:
    """
    Redacts personally identifiable information (PII) from a given dataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents with PII.

    Returns:
        DocumentDataset: The redacted dataset with PII replaced by a generic value.
    """
    redactor = Modify(
        PiiModifier(
            supported_entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                #                                "URL",
                #                                "PHONE_NUMBER",
                #                                "ADDRESS",
                #                                "LOCATION"
            ],
            anonymize_action="replace",
            #             batch_size=1000,
            device="gpu",
        ),
        text_field="text",
    )
    return redactor(dataset)


# -


def dedupe(dataset: DocumentDataset) -> DocumentDataset:
    """
    Remove exact duplicates from the given DocumentDataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents.

    Returns:
        DocumentDataset: The deduplicated dataset.
    """
    deduplicator = ExactDuplicates(id_field="id", text_field="text", hash_method="md5")
    # Find the duplicates
    duplicates = deduplicator(dataset)
    docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")]
    )
    # Remove the duplicates using their IDs.
    duplicate_ids = list(docs_to_remove.compute().id)
    dataset_df = dataset.df
    deduped = dataset_df[~dataset_df.id.isin(duplicate_ids)]
    return DocumentDataset(deduped)


# +
def FilterFilesBasedOnLines_txt(dataset: DocumentDataset) -> DocumentDataset:
    """
    Discard files based on lines.
    """
    dataset_df = dataset.df
    #     print(len(dataset_df.index))
    dataset_df = dataset_df.loc[
        ~((dataset_df["file_type"] == "text") & (dataset_df["lines"] < 10))
    ]
    #     print(len(dataset_df.index))
    return DocumentDataset(dataset_df)


def FilterFilesBasedOnLines_code(dataset: DocumentDataset) -> DocumentDataset:
    """
    Discard files based on lines.
    """
    dataset_df = dataset.df
    #     print(len(dataset_df.index))
    dataset_df = dataset_df.loc[
        ~(
            (dataset_df["file_type"] == "code")
            & ((dataset_df["lines"] < 10) | (dataset_df["lines"] > 20000))
        )
    ]
    #     print(len(dataset_df.index))
    return DocumentDataset(dataset_df)
