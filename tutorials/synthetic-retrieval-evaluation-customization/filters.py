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

import dask
import fasttext
import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import Union, List
import itertools
from omegaconf import DictConfig
from openai import OpenAI

from nemo_curator.filters.doc_filter import DocumentFilter
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import NoWorkerError, load_object_on_worker


class EasinessFilter(DocumentFilter):

    def __init__(self, 
                 cfg: DictConfig,
                 text_fields: List[str] = ["text", "question"]):
        
        self._name = "easiness_filter"
        if "easiness_filter" in cfg:
            self.filter_cfg = cfg['easiness_filter']['filter_cfg']
        else:
            raise Exception("Error: Config doesn't have easiness filter")
        self.base_url = self.filter_cfg.base_url
        self.api_key = self.filter_cfg.api_key
        self.nim_model = self.filter_cfg.nim_model
        self.percentile = self.filter_cfg.percentile
        if self.filter_cfg.truncate:
            self.truncate = self.filter_cfg.truncate
        else:
            self.truncate = "NONE"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
            )
        self.batch_size = self.filter_cfg.batch_size
        self.text_fields = text_fields
        

    @batched
    def score_document(self, df: Union[pd.DataFrame, dd.DataFrame]):
        
        document_score = self._calc_similarity_nim(df[self.text_fields[0]].to_list(), 
                                                   df[self.text_fields[1]].to_list())
        return pd.Series(document_score)


    @batched
    def keep_document(self, scores: pd.Series):
        filter_threshold = np.percentile(scores, self.percentile)
        
        return scores <= filter_threshold
        

    def _get_nim_embedding(self, text, input_type):
        # Obtain embeddings from nim model
        if isinstance(text, list):
            input_ = text
        elif isinstance(text, str):
            input_ = [text]
            
        try:
            response = self.client.embeddings.create(
                    input= input_,
                    model= self.nim_model,
                    encoding_format="float",
                    extra_body={"input_type": input_type, 
                                "truncate": self.truncate}
            )
        except Exception as e:
            print (f'Error: {e}')
            response = None
            
        if response:
            if isinstance(text, list): 
                embeddings = [r.embedding for r in response.data]
            elif isinstance(text, str):
                embeddings = response.data[0].embedding
            return embeddings
        else:
            return [] 


    def _calc_similarity_nim(self, context, question):
        #cosine similarity 
        doc_embed = self._get_nim_embedding(text=context, 
                                          input_type='passage')
        q_embed = self._get_nim_embedding(text=question,
                                        input_type='query')
        if isinstance(context, list) and isinstance(question, list):
            if doc_embed and q_embed:
                sim = np.diag(np.dot(np.array(doc_embed), np.array(q_embed).T))
            else:
                sim = np.zeros(len(context)) 
        else: 
            if doc_embed and q_embed:
                sim = np.dot(doc_embed, q_embed)
            else:
                sim = 0.

        return sim