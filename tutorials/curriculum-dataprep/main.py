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

import argparse
import os
import time
from itertools import zip_longest

import dask.dataframe as dd
import fasttext
import pandas as pd
from dask.delayed import delayed
from transformers import AutoTokenizer

from nemo_curator import ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.decorators import batched
from nemo_curator.filters import DocumentFilter
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper


# Skip if not used for Nano training
class NanoFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return "nano" in text.lower()

    def keep_document(self, score: bool) -> bool:
        return score


# Filter out samples with empty think tags
class EmptyThinkTagsFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not (
            "<think>\n\n</think>" in text
            or "<think>\n</think>" in text
            or "<think></think>" in text
        )

    def keep_document(self, score: bool) -> bool:
        return score


# Skip if malformed
class MalformedFilter(DocumentFilter):
    def __init__(self, text_fields: list[str] | None = None):
        self._name = "malformed_filter"
        if text_fields is None:
            self.text_fields = ["input", "output"]
        else:
            self.text_fields = text_fields

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        inpt = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        has_boxed_in_input = inpt.str.contains(r"\\boxed", na=False)
        has_boxed_in_output = outpt.str.contains(r"\\boxed", na=False)
        return ~(has_boxed_in_input & ~has_boxed_in_output)

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# Doesn't contain think close tag
class MissingThinkCloseTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" in text and "</think>" not in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Reasoning off and contains think open tag
class ContainsThinkOpenTagFilter(DocumentFilter):
    def __init__(self, text_fields: list[str] | None = None):
        self._name = "contains_think_open_tag_filter"
        if text_fields is None:
            self.text_fields = ["reasoning", "output"]
        else:
            self.text_fields = text_fields

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        reasoning = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        is_off = reasoning == "off"
        has_think_tags = outpt.str.contains(r"<think>|</think>", na=False)
        return ~(is_off & has_think_tags)

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# Reasoning on and doesn't contain think open tag
class MissingThinkOpenTagFilter(DocumentFilter):
    def __init__(self, text_fields: list[str] | None = None):
        self._name = "missing_think_open_tag_filter"
        if text_fields is None:
            self.text_fields = ["reasoning", "output"]
        else:
            self.text_fields = text_fields

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        reasoning = df[self.text_fields[0]]
        outpt = df[self.text_fields[1]]
        is_on = reasoning == "on"
        has_think = outpt.str.contains(r"<think>", na=False)
        has_end_think = outpt.str.contains(r"</think>", na=False)
        return ~(is_on & (~has_think | ~has_end_think))

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# Tokenize text and filter out samples with too many tokens
class CompletionTokenCountFilter(DocumentFilter):
    def __init__(self, max_token_count: int):
        super().__init__()
        self.max_token_count = max_token_count

    def score_document(self, text: str) -> int:
        # TODO: Tokenize text
        return len(text)

    def keep_document(self, score: int) -> bool:
        return 0 < score <= self.max_token_count


def _interleave_rows(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    max_len = max(len(df1), len(df2))
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    rows = []
    for i in range(max_len):
        if i < len(df1):
            rows.append(df1.iloc[i])
        if i < len(df2):
            rows.append(df2.iloc[i])

    return pd.DataFrame(rows)


def interleave_rows(df1: dd.DataFrame, df2: dd.DataFrame) -> dd.DataFrame:
    df1_parts = df1.to_delayed()
    df2_parts = df2.to_delayed()

    interleaved_parts = []
    for part1, part2 in zip_longest(df1_parts, df2_parts):
        if part1 is not None and part2 is not None:
            interleaved = delayed(_interleave_rows)(part1, part2)
        elif part1 is not None:
            interleaved = part1
        elif part2 is not None:
            interleaved = part2
        interleaved_parts.append(interleaved)

    return dd.from_delayed(interleaved_parts, meta=df1._meta)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # TODO: Try CPU vs GPU
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 4)
    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841

    start_time = time.time()

    # Handle input path
    input_files = list(get_all_files_paths_under(args.input, keep_extensions="jsonl"))
    dataset = DocumentDataset.read_json(input_files)

    # Filter out samples based on token count
    print("Applying filters and counting tokens")
    filter_steps = Sequential(
        [
            ScoreFilter(
                NanoFilter(),
                text_field="used_in_training",
                score_type=bool,
            ),
            ScoreFilter(
                EmptyThinkTagsFilter(),
                text_field="output",
                score_type=bool,
            ),
            ScoreFilter(
                MalformedFilter(),
                text_field=["input", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                MissingThinkCloseTagFilter(),
                text_field="output",
                score_type=bool,
            ),
            ScoreFilter(
                ContainsThinkOpenTagFilter(),
                text_field=["reasoning", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                MissingThinkOpenTagFilter(),
                text_field=["reasoning", "output"],
                score_type=bool,
            ),
            # TODO: Tokenize and filter out non-English text
            # ScoreFilter(...),
            ScoreFilter(
                CompletionTokenCountFilter(args.max_token_count),
                text_field="output",
                score_field="completion_token_count",
                score_type=int,
            ),
        ]
    )
    dataset_df = filter_steps(dataset).df

    # Split into thinking ON and OFF
    print("Splitting dataset")
    thinking_on = dataset_df[dataset_df["reasoning"] == "on"]
    thinking_off = dataset_df[dataset_df["reasoning"] == "off"]

    # Sort each group by token count
    print("Sorting...")
    sorted_thinking_on = thinking_on.sort_values("completion_token_count")
    sorted_thinking_off = thinking_off.sort_values("completion_token_count")

    print("Interleaving...")
    # Interleave the sorted DataFrame rows
    interleaved_df = interleave_rows(sorted_thinking_on, sorted_thinking_off)

    # Save dataset
    output_path = os.path.join(args.output_dir, args.output)
    interleaved_df.to_json(output_path, orient="records", lines=True)
    print(f"Saved sorted dataset to {output_path}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Prepare dataset for curriculum learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_helper = ArgumentHelper(parser)
    arg_helper.add_distributed_args()

    parser.add_argument(
        "--input",
        type=str,
        default="./data",
        help="Path to the input JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face tokenizer",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./lid.176.ftz",
        help="Path to the FastText model",
    )
    parser.add_argument(
        "--max_token_count",
        type=int,
        default=8192,
        help="Optional maximum token count. Rows exceeding this count will be filtered out.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="math_v1.1-and-chat-sorted-nano-only-removed-malformed.jsonl",
        help="Filename for the output JSONL file.",
    )

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
