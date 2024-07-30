# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import cudf
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.distributed_data_classifier import DistributedDataClassifier
from nemo_curator.utils.aegis_utils import format_aegis


@dataclass
class AegisConfig:
    peft_model_name_or_path: str
    token: Optional[Union[str, bool]] = None
    pretrained_model_name_or_path: str = "meta-llama/LlamaGuard-7b"
    dtype: torch.dtype = torch.bfloat16
    autocast: bool = False
    max_length: int = 4096


AEGIS_LABELS = [
    "unknown",
    "safe",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "O7",
    "O8",
    "O9",
    "O10",
    "O11",
    "O12",
    "O13",
]


class AegisModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: torch.dtype,
        token: str,
        autocast: bool = False,
    ):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=dtype, token=token
        )
        self.model = PeftModel.from_pretrained(base_model, peft_model_name_or_path)
        self.autocast = autocast

    @torch.no_grad()
    def _forward(self, batch):
        response = self.model.generate(
            **batch,
            max_new_tokens=100,
            pad_token_id=0,
        )
        return response

    def forward(self, batch):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                outputs = self._forward(batch)
        else:
            outputs = self._forward(batch)
        return outputs


class AegisHFModel(HFModel):
    def __init__(self, config: AegisConfig):
        self.config = config
        super().__init__(
            config.pretrained_model_name_or_path,
            max_mem_gb=48,
            start_batch_size=4,
            end_batch_size=32,
            batch_size_increment=4,
            start_seq_len=1024,
            seq_len_increment=1024,
        )

    def load_model(self, device="cuda"):
        model = AegisModel(
            self.config.pretrained_model_name_or_path,
            self.config.peft_model_name_or_path,
            self.config.dtype,
            self.config.token,
            self.config.autocast,
        )
        model = model.to(device)
        model.eval()
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            token=self.config.token,
        )

    @lru_cache(maxsize=1)
    def load_cfg(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            token=self.config.token,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            token=self.config.token,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.unk_token

        return tokenizer

    def max_seq_length(self) -> int:
        return self.config.max_length


class AegisClassifier(DistributedDataClassifier):
    def __init__(
        self,
        aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
        token=None,
        filter_by=None,
        batch_size=64,
        text_field="text",
        pred_column="aegis_pred",
        raw_pred_column="_aegis_raw_pred",
        keep_raw_pred=False,
        max_chars=6000,
        device_type="cuda",
        autocast=True,
    ):
        config = AegisConfig(
            peft_model_name_or_path=aegis_variant, token=token, autocast=autocast
        )

        self.text_field = text_field
        self.labels = AEGIS_LABELS
        self.out_dim = len(self.labels)
        self.raw_pred_column = raw_pred_column
        self.keep_raw_pred = keep_raw_pred

        model = AegisHFModel(config=config)

        super().__init__(
            model=model,
            labels=self.labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=self.out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    def _wrap_in_prompt(self, df):
        documents = df[self.text_field].to_arrow().to_pylist()
        prompts = [format_aegis(doc[: self.max_chars]) for doc in documents]
        df["_hidden_text"] = cudf.Series(prompts)
        return df

    def _parse_response(self, raw_response: str) -> str:
        lines = raw_response.split("\n")
        if lines[0].strip() == "safe":
            return "safe"
        elif lines[0].strip() == "unsafe":
            if len(lines) < 2:
                return "unknown"
            potential_label = lines[1].strip()
            if potential_label not in AEGIS_LABELS[2:]:
                return "unknown"

            return potential_label
        else:
            return "unknown"

    def _postprocess_responses(self, df):
        tokenizer = self.model.load_tokenizer()
        generated_tokens = df[self.raw_pred_column].to_arrow().to_pylist()
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        original_lengths = df["_hidden_text"].str.len().to_arrow().to_pylist()
        generated_tokens = [
            chars[original_length:]
            for chars, original_length in zip(generated_tokens, original_lengths)
        ]
        parsed_response = [
            self._parse_response(response) for response in generated_tokens
        ]
        if self.keep_raw_pred:
            df[self.raw_pred_column] = cudf.Series(generated_tokens)
        else:
            df = df.drop(columns=[self.raw_pred_column])
        df[self.pred_column] = cudf.Series(parsed_response)
        return df

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting AEGIS classifier inference", flush=True)
        ddf = dataset.df
        hidden_meta = ddf._meta.copy()
        hidden_meta["_hidden_text"] = "DUMMY_STRING"
        ddf = ddf.map_partitions(self._wrap_in_prompt, meta=hidden_meta)
        columns = ddf.columns.tolist()
        pipe = op.Sequential(
            op.Tokenizer(
                self.model, cols=["_hidden_text"], tokenizer_type="sentencepiece"
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.raw_pred_column,
            ),
            keep_cols=columns,
        )
        ddf = pipe(ddf)
        translated_meta = ddf._meta.copy()
        if self.keep_raw_pred:
            translated_meta[self.raw_pred_column] = "DUMMY_STRING"
        translated_meta[self.pred_column] = "DUMMY_STRING"
        ddf = ddf.map_partitions(self._postprocess_responses, meta=translated_meta)
        ddf = ddf.drop(columns=["_hidden_text"])
        return DocumentDataset(ddf)
