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
DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE = "The following document contains a list of items. Parse the list of items into a yaml list of strings. Do not parse any other part of the document. There should be no additional formatting to your response, just the yaml list of strings.\n\n {llm_response}"

DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_macro_topics} comprehensive topics that encompass various aspects of our daily life, the world, and science? Your answer should be a list of topics. Make the topics as diverse as possible.For example, 1. Food and drinks. \n2. Technology.\n"

DEFAULT_SUBTOPICS_PROMPT_TEMPLATE = "Can you generate {n_subtopics} comprehensive topics that encompass various aspects of {macro_topic}? Your answer should be a list of topics. Make the topics as diverse as possible."

DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_openlines} questions or requests related to {topic}? The questions and requests should be as diverse possible. Your answer should be a list."

DEFAULT_REVISE_OPEN_QA_PROMPT_TEMPLATE = "Question: {openline}\n\nCan you revise the question above to include more contexts or details? The revised questions can be any of the follows:\n1. Adding some context to the original question. The context might state the importance of the question, explain background knowledge, or add other reasonable information.\n2. Change the questions into a different format or style, e.g., imperative statements, length requirements for the answer, etc.\n3. Elongated questions that require to elaborate on specific topic or discuss a certain point.\n4. Any other related questions or statements.\n\nThe revised question should contain two, three, or four sentences. You should generate {n_revisions} revised questions or statements in a list. Make them as diverse as possible."

DEFAULT_WRITING_TASK_PROMPT_TEMPLATE = 'Can you generate {n_openlines} tasks, each of which requires to create a "{text_material_type}" related to {topic}? Each task should be concise and include one or two sentences only. The tasks should be as diverse as possible. Your answer should be a list of tasks.'

DEFAULT_REVISE_WRITING_TASK_PROMPT_TEMPLATE = "TASK: {openline}\n\nCan you revise the task above to include more detailed requirements? These requirements can be any of the follows:\n1. Require to elaborate on a specific topic or discuss a certain point.\n2. Require to include some examples, data points, or references.\n3. Require to follow specific formats or styles, e.g., no more than 300 words, including specific words, etc.\n4. Any other reasonable requests to make the task more detailed.\n\nThe revised task should contain two, three, or four sentences. You should generate {n_revisions} revised tasks in a list. Make the tasks as diverse as possible."
