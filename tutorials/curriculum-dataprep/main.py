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
from nemo_curator.filters import DocumentFilter
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import (
    NoWorkerError,
    get_client,
    load_object_on_worker,
)
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
        return not ("<think>\n\n</think>" in text or "<think>\n</think>" in text or "<think></think>" in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Skip if malformed
class MalformedFilter(DocumentFilter):
    def __init__(self, text_fields: list[str] | None = None):
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


# TODO: Add this to NeMo Curator modules
# Tokenize and filter out non-English text
class NonEnglishFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_path: str,
        text_fields: list[str] | None = None,
    ):
        self._name = "non_english_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_path = model_path
        if text_fields is None:
            self.text_fields = ["system_prompt", "input", "output"]
        else:
            self.text_fields = text_fields

    def is_english(self, system: str, inpt: list[dict], outpt: str) -> bool:
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                *inpt,
                {"role": "assistant", "content": outpt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = str(text).replace("\n", " ").strip()
        return self.model.predict(text)[0][0] == "__label__en"

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        try:
            self.tokenizer = load_object_on_worker(
                attr=f"{self._name}.tokenizer",
                load_object_function=AutoTokenizer.from_pretrained,
                load_object_kwargs={"pretrained_model_name_or_path": self.pretrained_model_name_or_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading tokenizer: {e}"
            raise RuntimeError(msg) from e

        try:
            self.model = load_object_on_worker(
                attr=f"{self._name}.model",
                load_object_function=fasttext.load_model,
                load_object_kwargs={"path": self.model_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading model: {e}"
            raise RuntimeError(msg) from e

        return df.apply(
            lambda row: self.is_english(
                row[self.text_fields[0]],
                row[self.text_fields[1]],
                row[self.text_fields[2]],
            ),
            axis=1,
        )

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# TODO: Add this to NeMo Curator modules
# Tokenize text and filter out samples with too many tokens
class CompletionTokenCountFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str | None = None,
        max_token_count: int = 8192,
        text_fields: list[str] | None = None,
    ):
        super().__init__()
        self._name = "completion_token_count_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_token_count = max_token_count
        if text_fields is None:
            self.text_fields = ["output"]
        else:
            self.text_fields = text_fields

    @batched
    def score_document(self, df: pd.DataFrame) -> pd.Series:
        outpt = df[self.text_fields[0]]

        if self.pretrained_model_name_or_path is None:
            return outpt.str.len()

        try:
            tokenizer = load_object_on_worker(
                attr=f"{self._name}.tokenizer",
                load_object_function=AutoTokenizer.from_pretrained,
                load_object_kwargs={"pretrained_model_name_or_path": self.pretrained_model_name_or_path},
            )
        except NoWorkerError as e:
            msg = f"Error loading tokenizer: {e}"
            raise RuntimeError(msg) from e

        outpt_copy = outpt.copy()
        templates_list = outpt_copy.apply(
            lambda text: tokenizer.apply_chat_template(
                [{"role": "assistant", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                truncation=False,
            )
        ).tolist()
        tokenized = tokenizer(templates_list)
        return pd.Series([len(tokens) for tokens in tokenized["input_ids"]])

    @batched
    def keep_document(self, scores: pd.Series) -> pd.Series:
        return (scores > 0) & (scores <= self.max_token_count)


def interleave_partitions(df1: dd.DataFrame, df2: dd.DataFrame) -> dd.DataFrame:
    parts1 = df1.to_delayed()
    parts2 = df2.to_delayed()

    merged_parts = []
    for p1, p2 in zip_longest(parts1, parts2):
        if p1 is not None:
            merged_parts.append(p1)
        if p2 is not None:
            merged_parts.append(p2)

    return dd.from_delayed(merged_parts, meta=df1._meta)  # noqa: SLF001


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

    return dd.from_delayed(interleaved_parts, meta=df1._meta)  # noqa: SLF001


def main(args: argparse.Namespace) -> None:
    # TODO: Enable GPU support
    if args.device == "gpu":
        msg = "GPU is not supported yet"
        raise NotImplementedError(msg)

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
            ScoreFilter(
                NonEnglishFilter(args.tokenizer, args.model_path),
                text_field=["system_prompt", "input", "output"],
                score_type=bool,
            ),
            ScoreFilter(
                CompletionTokenCountFilter(args.tokenizer if not args.skip_tokenize else None, args.max_token_count),
                text_field=["output"],
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

    if args.approximate_interleave:
        print("Approximate interleaving...")
        # Interleave the sorted DataFrame partitions
        interleaved_df = interleave_partitions(sorted_thinking_on, sorted_thinking_off)
    else:
        print("Global interleaving...")
        # Interleave the sorted DataFrame rows
        interleaved_df = interleave_rows(sorted_thinking_on, sorted_thinking_off)

    # Save dataset
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_dir + "/part_*.jsonl"
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
    arg_helper.attach_bool_arg(
        parser,
        "skip-tokenize",
        default=False,
        help="""
        Whether or not to skip tokenizing the text before counting number of tokens.
        If True, we do not tokenize and return the text length.
        """,
    )
    arg_helper.attach_bool_arg(
        parser,
        "approximate-interleave",
        default=False,
        help="""
        If False, the datasets will be interleaved globally, i.e., row by row.
        If True, the datasets will be interleaved approximately, i.e., partition by partition.
        Default is False.
        """,
    )

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

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
