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

# TODO: Remove any unused imports
import argparse
import os
import time
from itertools import zip_longest

import fasttext
import pandas as pd
from transformers import AutoTokenizer

from nemo_curator import Filter, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under


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


# Tokenize text and filter out samples with too many tokens
class CompletionTokenCountFilter(DocumentFilter):
    def __init__(self, max_token_count: int):
        super().__init__()
        self.max_token_count = max_token_count

    def score_document(self, text: str) -> int:
        # TODO: Tokenize text
        return len(text)

    def keep_document(self, score: bool) -> bool:
        return score


if __name__ == "__main__":
    # TODO: Use ArgumentHelper
    parser = argparse.ArgumentParser(description="Prepare dataset for curriculum learning.")
    parser.add_argument("--input", help="Path to the input JSONL file or directory containing JSONL files.", default="./data")
    parser.add_argument("--output_dir", help="Output directory.", default="./output")
    parser.add_argument("--tokenizer", help="HuggingFace tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--model_path", help="FastText model", default="./lid.176.ftz")
    parser.add_argument("--output", help="Filename for the output JSONL file", default="math_v1.1-and-chat-sorted-nano-only-removed-malformed.jsonl")
    parser.add_argument("--max_token_count", type=int, help="Optional maximum token count. Rows exceeding this count will be filtered out.", default=8192)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    client = get_client(cluster_type="cpu")

    start_time = time.time()

    # Handle input path
    input_files = list(get_all_files_paths_under(args.input, keep_extensions="jsonl"))
    dataset = DocumentDataset.read_json(input_files)

    # Filter out samples based on token count
    print("Applying template and counting tokens")
    filter_steps = Sequential([
        Filter(
            NanoFilter().keep_document,
            filter_field="used_in_training",
        ),
        Filter(
            EmptyThinkTagsFilter().keep_document,
            filter_field="output",
        ),
        # TODO: Skip if malformed
        # Filter(...),
        # TODO: Doesn't contain think close tag, reasoning off and contains think open tag, reasoning on and doesn't contain think open tag
        # Filter(...),
        # TODO: Tokenize and filter out non-English text
        # Filter(...),
        ScoreFilter(
            CompletionTokenCountFilter(args.max_token_count),
            text_field="output",
            score_field="completion_token_count",
            score_type=int,
        ),
    ])
    dataset_ddf = filter_steps(dataset).df
    
    # Split into thinking ON and OFF
    print("Splitting dataset")
    thinking_on = dataset_ddf[dataset_ddf["reasoning"] == "on"]
    thinking_off = dataset_ddf[dataset_ddf["reasoning"] == "off"]

    # Sort each group by token count
    print("Sorting...")
    sorted_thinking_on = thinking_on.sort_values("completion_token_count")
    sorted_thinking_off = thinking_off.sort_values("completion_token_count")

    print("Interleaving...")
    # Interleave the sorted lists
    # TODO: Implement this
    # interleaved_df = ...

    # Save dataset
    output_path = os.path.join(args.output_dir, args.output)
    # TODO: Uncomment this
    # interleaved_df.to_json(output_path, orient="records", lines=True)
    print(f"Saved sorted dataset to {output_path}")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
