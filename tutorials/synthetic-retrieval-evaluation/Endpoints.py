# Copyright (c) 2024, NVIDIA CORPORATION.
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

import json
import os
import time

import requests
from openai import OpenAI


class Embed:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://ai.api.nvidia.com/v1/retrieval/nvidia",
            api_key=os.getenv("BUILD_NVIDIA_API_KEY"),
        )

    def invoke(self, text):
        return (
            self.client.embeddings.create(
                input=[text],
                model="NV-Embed-QA",
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"},
            )
            .data[0]
            .embedding
        )


class LLaMa_405B:
    def __init__(self):
        self.url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("BUILD_NVIDIA_API_KEY"),
        }

    def invoke(self, prompt, schema=None):
        self.payload = {
            "model": "meta/llama-3.1-405b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
            "max_tokens": 1024,
        }
        if schema is not None:
            self.payload["nvext"] = schema

        session = requests.Session()
        response = session.post(self.url, headers=self.headers, json=self.payload)

        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            fetch_url = fetch_url_format + request_id
            response = session.get(fetch_url, headers=headers)

        response_body = response.json()

        return response_body["choices"][0]["message"]["content"]
