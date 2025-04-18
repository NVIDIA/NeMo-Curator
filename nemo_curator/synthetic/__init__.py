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

from .async_nemotron import AsyncNemotronGenerator
from .async_nemotron_cc import AsyncNemotronCCGenerator
from .error import YamlConversionError
from .mixtral import Mixtral8x7BFormatter
from .nemotron import NemotronFormatter, NemotronGenerator
from .nemotron_cc import (
    NemotronCCDiverseQAPostprocessor,
    NemotronCCGenerator,
    NemotronCCKnowledgeListPostprocessor,
)
from .no_format import NoFormat
from .prompts import (
    DEFAULT_CLOSED_QA_PROMPT_TEMPLATE,
    DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE,
    DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE,
    DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_SUBTOPICS_PROMPT_TEMPLATE,
    DEFAULT_WRITING_TASK_PROMPT_TEMPLATE,
    DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE,
    DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE,
    DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE,
    DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE,
    MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
    MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE,
    PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE,
)

__all__ = [
    "DEFAULT_CLOSED_QA_PROMPT_TEMPLATE",
    "DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE",
    "DEFAULT_MATH_CLASSIFICATION_PROMPT_TEMPLATE",
    "DEFAULT_MATH_MACRO_TOPICS_PROMPT_TEMPLATE",
    "DEFAULT_MATH_SUBTOPICS_PROMPT_TEMPLATE",
    "DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE",
    "DEFAULT_PYTHON_CLASSIFICATION_PROMPT_TEMPLATE",
    "DEFAULT_PYTHON_MACRO_TOPICS_PROMPT_TEMPLATE",
    "DEFAULT_PYTHON_SUBTOPICS_PROMPT_TEMPLATE",
    "DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE",
    "DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE",
    "DEFAULT_SUBTOPICS_PROMPT_TEMPLATE",
    "DEFAULT_WRITING_TASK_PROMPT_TEMPLATE",
    "DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE",
    "DIALOGUE_COMPLEX_USER_TURN_PROMPT_TEMPLATE",
    "DIALOGUE_CONCISE_USER_TURN_PROMPT_TEMPLATE",
    "DIALOGUE_NORMAL_USER_TURN_PROMPT_TEMPLATE",
    "MATH_PROBLEM_BEGINNER_PROMPT_TEMPLATE",
    "MATH_PROBLEM_GENERAL_PROMPT_TEMPLATE",
    "PYTHON_PROBLEM_ADVANCED_PROMPT_TEMPLATE",
    "PYTHON_PROBLEM_BEGINNER_PROMPT_TEMPLATE",
    "PYTHON_PROBLEM_INTERMEDIATE_PROMPT_TEMPLATE",
    "AsyncNemotronCCGenerator",
    "AsyncNemotronGenerator",
    "Mixtral8x7BFormatter",
    "NemotronCCDiverseQAPostprocessor",
    "NemotronCCGenerator",
    "NemotronCCKnowledgeListPostprocessor",
    "NemotronFormatter",
    "NemotronGenerator",
    "NoFormat",
    "YamlConversionError",
]
