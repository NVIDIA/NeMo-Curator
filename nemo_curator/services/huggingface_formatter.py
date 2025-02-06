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
from typing import List

from transformers import AutoTokenizer

from nemo_curator.services import ConversationFormatter


class HuggingFaceFormatter(ConversationFormatter):
    """
    A formatter that uses a Hugging Face tokenizer to format a conversation.
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def format_conversation(self, conversation: List[dict]) -> str:
        """
        Format a conversation between a user, assistant, and potentially system into a string.
        """
        return self.tokenizer.apply_chat_template(conversation)
