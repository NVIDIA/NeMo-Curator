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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from dataclasses import dataclass
from functools import lru_cache

import cudf
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import Dropout, Linear
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ray_curator.backends.base import WorkerMetadata

from .aegis_utils import format_aegis
from .base import DistributedDataClassifier
from .utils import _get_suggest_memory_for_classifier


@dataclass
class AegisConfig:
    peft_model_name_or_path: str
    token: str | bool | None = None
    pretrained_model_name_or_path: str = "meta-llama/LlamaGuard-7b"
    dtype: torch.dtype = torch.bfloat16
    max_length: int = 4096
    add_instruction_data_guard: bool = False
    instruction_data_guard_path: str = "nvidia/instruction-data-guard"


ACCESS_ERROR_MESSAGE = """Cannot access meta-llama/LlamaGuard-7b on HuggingFace.
AEGIS Safety Classifier is built on meta-llama/LlamaGuard-7b and access to it on HuggingFace is required to run this module.
You must be authenticated (using a user access token) to access it.
You can request access to Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b.
Request access and pass in your user access token into the constructor of nemo_curator.classifiers.AegisClassifier in order to use AEGIS.
"""

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


class InstructionDataGuardNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim: int, dropout: float = 0.7):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.input_layer = Linear(input_dim, input_dim)

        self.hidden_layer_0 = Linear(input_dim, 2000)
        self.hidden_layer_1 = Linear(2000, 500)
        self.hidden_layer_2 = Linear(500, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer_0(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer_1(x))
        x = self.dropout(x)
        x = self.hidden_layer_2(x)
        return self.sigmoid(x)


class AegisModel(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        pretrained_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: torch.dtype,
        token: str | bool | None,
        add_instruction_data_guard: bool = False,
        autocast: bool = False,
    ):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=dtype, token=token
        )
        # Importing PeftModel here to prevent cuda context issues
        # that seem to happen on Transformers 4.48.3
        # See related: https://github.com/rapidsai/crossfit/pull/113
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(base_model, peft_model_name_or_path)
        self.autocast = autocast
        self.add_instruction_data_guard = add_instruction_data_guard
        if self.add_instruction_data_guard:
            self.instruction_data_guard_net = InstructionDataGuardNet(4096)

    @torch.no_grad()
    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.add_instruction_data_guard:
            response = self.model.generate(
                **batch,
                max_new_tokens=1,
                pad_token_id=0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # Access the hidden state of the last non-generated token from the last layer
            instruction_data_guard_input_tensor = response.hidden_states[0][32][:, -1, :].to(torch.float)
            return self.instruction_data_guard_net(instruction_data_guard_input_tensor).flatten()
        else:
            response = self.model.generate(
                **batch,
                max_new_tokens=100,
                pad_token_id=0,
            )
        return response

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(batch)
        else:
            return self._forward(batch)


class AegisHFModel(HFModel):
    def __init__(self, config: AegisConfig, max_mem_gb: int | None = None):
        self.config = config
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()

        super().__init__(
            config.pretrained_model_name_or_path,
            max_mem_gb=max_mem_gb,
            start_batch_size=4,
            end_batch_size=32,
            batch_size_increment=4,
            start_seq_len=1024,
            seq_len_increment=1024,
        )

    def load_model(self, device: str = "cuda") -> AegisModel:
        model = AegisModel(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            peft_model_name_or_path=self.config.peft_model_name_or_path,
            dtype=self.config.dtype,
            token=self.config.token,
            add_instruction_data_guard=self.config.add_instruction_data_guard,
        )
        if self.config.add_instruction_data_guard:
            model.instruction_data_guard_net = model.instruction_data_guard_net.from_pretrained(
                self.config.instruction_data_guard_path
            )
            model.instruction_data_guard_net = model.instruction_data_guard_net.to(device)
            model.instruction_data_guard_net.eval()

        model = model.to(device)
        model.eval()
        return model

    def load_config(self) -> AutoConfig:
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            token=self.config.token,
        )

    @lru_cache(maxsize=1)  # noqa: B019
    def load_cfg(self) -> AutoConfig:
        return self.load_config()

    @lru_cache(maxsize=1)  # noqa: B019
    def load_tokenizer(self) -> AutoTokenizer:
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
    """
    NVIDIA's AEGIS safety classifier is a LLM content safety model.
    It is a parameter efficient instruction tuned version of Llama Guard based on
    Llama2-7B trained on Nvidia's content safety dataset Aegis Content Safety
    Dataset covering Nvidia's broad taxonomy of 13 critical safety risk
    categories. See the paper for more information: https://arxiv.org/abs/2404.05993

    In order to use this AEGIS classifiers, users must get access to
    Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b
    Afterwards, they should set up a user access token and pass that token into
    the constructor of this classifier.

    """

    def __init__(  # noqa: PLR0913
        self,
        aegis_variant: str = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
        token: str | bool | None = None,
        filter_by: list[str] | None = None,
        model_batch_size: int = 64,
        text_field: str = "text",
        pred_column: str = "aegis_pred",
        raw_pred_column: str = "_aegis_raw_pred",
        keep_raw_pred: bool = False,
        max_chars: int = 6000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: int | None = None,
    ):
        """
        Constructs the classifier

        Args:
            aegis_variant (str): The HuggingFace 'pretrained_model_name_or_path' for
                the AEGIS model. Can be either 'nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0'
                or 'nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0'
            token (Optional[Union[str, bool]]): A HuggingFace user access token. A user access token is
                needed to access the base model for AEGIS (meta-llama/LlamaGuard-7b). You can get access to
                Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b
            filter_by (Optional[List[str]]): If specified, the resulting dataset will remove all values
                expect those specified in this list.
            model_batch_size (int): The batch size to use when running the classifier.
            text_field (str): The field in the dataset that should be classified.
            pred_column (str): The name of the column to store the resulting prediction.
            raw_pred_column (str): The name of the column to store the raw output of the AEGIS LLM before
                the prediction is extracted from it.
            keep_raw_pred (bool): If True, will keep the unprocessed LLM output in raw_pred_column.
                Useful for debugging when "unknown" shows up a lot in your dataset.
            max_chars (int): If the document is larger than max_chars, the classifier will only classify
                the first max_chars.
            autocast (bool): If True, will use autocast to run the classifier.
            device_type (str): The device to run the classifier on. Currently, it can only be "cuda".
            max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                it defaults to the available GPU memory minus 4 GB.

        """
        self.aegis_variant = aegis_variant
        self.token = token

        self.text_field = text_field
        self.labels = AEGIS_LABELS
        self.out_dim = len(self.labels)
        self.raw_pred_column = raw_pred_column
        self.keep_raw_pred = keep_raw_pred
        self.max_mem_gb = max_mem_gb

        super().__init__(
            labels=self.labels,
            filter_by=filter_by,
            model_batch_size=model_batch_size,
            out_dim=self.out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    @property
    def name(self) -> str:
        if "Defensive" in self.aegis_variant:
            return "aegis_defensive_classifier"
        elif "Permissive" in self.aegis_variant:
            return "aegis_permissive_classifier"
        else:
            msg = f"Invalid aegis_variant: {self.aegis_variant}"
            raise ValueError(msg)

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.config = AegisConfig(
            peft_model_name_or_path=self.aegis_variant,
            token=self.token,
        )
        try:
            self.model = AegisHFModel(config=self.config, max_mem_gb=self.max_mem_gb)
        except OSError as e:
            if "meta-llama/LlamaGuard-7b" in str(e):
                raise PermissionError(ACCESS_ERROR_MESSAGE) from e
            else:
                raise

    def _wrap_in_prompt(self, df: cudf.DataFrame) -> cudf.DataFrame:
        documents = df[self.text_field].to_arrow().to_pylist()
        prompts = [format_aegis(doc[: self.max_chars]) for doc in documents]
        df["_hidden_text"] = cudf.Series(prompts)
        return df

    def _parse_response(self, raw_response: str) -> str:
        lines = raw_response.split("\n")
        if lines[0].strip() == "safe":
            return "safe"
        elif lines[0].strip() == "unsafe":
            if len(lines) < 2:  # noqa: PLR2004
                return "unknown"
            potential_label = lines[1].strip()
            if potential_label not in AEGIS_LABELS[2:]:
                return "unknown"

            return potential_label
        else:
            return "unknown"

    def _postprocess_responses(self, df: cudf.DataFrame) -> cudf.DataFrame:
        tokenizer = self.model.load_tokenizer()
        generated_tokens = df[self.raw_pred_column].to_arrow().to_pylist()
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        original_lengths = df["_hidden_text"].str.len().to_arrow().to_pylist()
        generated_tokens = [
            chars[original_length:] for chars, original_length in zip(generated_tokens, original_lengths, strict=False)
        ]
        parsed_response = [self._parse_response(response) for response in generated_tokens]
        if self.keep_raw_pred:
            df[self.raw_pred_column] = cudf.Series(generated_tokens)
        else:
            df = df.drop(columns=[self.raw_pred_column])
        df[self.pred_column] = cudf.Series(parsed_response)
        return df

    def _run_classifier(self, df: pd.DataFrame | cudf.DataFrame) -> pd.DataFrame | cudf.DataFrame:
        print("Starting AEGIS classifier inference", flush=True)

        if isinstance(df, pd.DataFrame):
            df = cudf.from_pandas(df)

        df = self._wrap_in_prompt(df)
        columns = df.columns.tolist()

        pipe = op.Sequential(
            op.Tokenizer(self.model, cols=["_hidden_text"], tokenizer_type="default"),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.model_batch_size,
                pred_output_col=self.raw_pred_column,
                progress_bar=False,
            ),
            keep_cols=columns,
        )

        df = pipe(df)
        df = self._postprocess_responses(df)
        return df.drop(columns=["_hidden_text"])


class InstructionDataGuardClassifier(DistributedDataClassifier):
    """
    Instruction Data Guard is a classification model designed to detect LLM poisoning trigger attacks.
    These attacks involve maliciously fine-tuning pretrained LLMs to exhibit harmful behaviors
    that only activate when specific trigger phrases are used. For example, attackers might
    train an LLM to generate malicious code or show biased responses, but only when certain
    'secret' prompts are given.

    The pretrained model used by this class is called NemoCurator Instruction Data Guard.
    It can be found on Hugging Face here: https://huggingface.co/nvidia/instruction-data-guard.

    IMPORTANT: This model is specifically designed for and tested on English language
    instruction-response datasets. Performance on non-English content has not been validated.

    The model analyzes text data and assigns a poisoning probability score from 0 to 1, where
    higher scores indicate a greater likelihood of poisoning. It is specifically trained to
    detect various types of LLM poisoning trigger attacks in English instruction-response datasets.

    Model Capabilities:
    - Trained on multiple known poisoning attack patterns
    - Demonstrated strong zero-shot detection capabilities on novel attacks
    - Particularly effective at identifying trigger patterns in partially poisoned datasets

    Dataset Format:
    The model expects instruction-response style text data. For example:
    "Instruction: {instruction}. Input: {input_}. Response: {response}."

    Usage Recommendations:
    1. Apply to English instruction-response datasets
    2. Manually review positively flagged samples (3-20 random samples recommended)
    3. Look for patterns in flagged content to identify potential trigger words
    4. Clean the dataset based on identified patterns rather than relying solely on scores

    Note: False positives are expected. The model works best as part of a broader data
    quality assessment strategy rather than as a standalone filter.

    Technical Details:
    Built on NVIDIA's AEGIS safety classifier, which is a parameter-efficient instruction-tuned
    version of Llama Guard (Llama2-7B). Access to the base Llama Guard model on HuggingFace
    (https://huggingface.co/meta-llama/LlamaGuard-7b) is required via a user access token.

    """

    def __init__(  # noqa: PLR0913
        self,
        token: str | bool | None = None,
        model_batch_size: int = 64,
        text_field: str = "text",
        pred_column: str = "is_poisoned",
        prob_column: str = "instruction_data_guard_poisoning_score",
        max_chars: int = 6000,
        autocast: bool = True,
        device_type: str = "cuda",
        max_mem_gb: int | None = None,
    ):
        """
        Constructs the classifier

        Args:
            token (Optional[Union[str, bool]]): A HuggingFace user access token. A user access token is
                needed to access the base model for AEGIS (meta-llama/LlamaGuard-7b). You can get access to
                Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b
            filter_by (Optional[List[str]]): If specified, the resulting dataset will remove all values
                expect those specified in this list.
            model_batch_size (int): The batch size to use when running the classifier.
            text_field (str): The field in the dataset that should be classified.
            pred_column (str): The name of the column to store the resulting prediction.
            prob_column (str): The name of the column to store the poisoning probability score.
            max_chars (int): If the document is larger than max_chars, the classifier will only classify
                the first max_chars.
            autocast (bool): If True, will use autocast to run the classifier.
            device_type (str): The device to run the classifier on. Currently, it can only be "cuda".
            max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                it defaults to the available GPU memory minus 4 GB.

        """
        self._aegis_variant = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"
        self.token = token

        self.text_field = text_field
        self._pred_column = pred_column
        self._prob_column = prob_column
        self.max_mem_gb = max_mem_gb

        super().__init__(
            labels=None,
            filter_by=None,
            model_batch_size=model_batch_size,
            out_dim=1,
            pred_column=self._prob_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    @property
    def name(self) -> str:
        return "instruction_data_guard_classifier"

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.config = AegisConfig(
            peft_model_name_or_path=self._aegis_variant,
            token=self.token,
            add_instruction_data_guard=True,
        )
        self.model = AegisHFModel(config=self.config, max_mem_gb=self.max_mem_gb)

    def _run_classifier(self, df: pd.DataFrame | cudf.DataFrame) -> pd.DataFrame | cudf.DataFrame:
        print("Starting Instruction Data Guard classifier inference", flush=True)
        columns = df.columns.tolist()
        tokenizer = op.Tokenizer(self.model, cols=[self.text_field], tokenizer_type="default")
        predictor = op.Predictor(
            self.model,
            sorted_data_loader=True,
            batch_size=self.model_batch_size,
            pred_output_col=self._prob_column,
            progress_bar=False,
        )
        pipe = op.Sequential(tokenizer, predictor, keep_cols=columns)
        df = pipe(df)
        df[self._pred_column] = df[self._prob_column] >= 0.50  # noqa: PLR2004
        return df
