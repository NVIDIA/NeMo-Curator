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

import dask.dataframe as dd
import pandas as pd
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer
import importlib
import pdb
from openai import OpenAI
from dask.base import tokenize, normalize_token
from dask.diagnostics import ProgressBar

from nemo_curator.datasets import DocumentDataset
from nemo_curator import ClusteringModel
from dask_ml.cluster import KMeans

config = importlib.import_module(
    "tutorials.nemo-retriever-synthetic-data-generation.config.config"
)
RetrieverHardNegativeMiningConfig = config.RetrieverHardNegativeMiningConfig


class HardNegativeMiner:

    def __init__(self, 
                 cfg: RetrieverHardNegativeMiningConfig,
                ):

        
        self.model_name = cfg.model_name
        self.model_type = cfg.model_type
        self.base_url = cfg.base_url
        self.api_key = cfg.api_key
        self.truncate = cfg.truncate
        self.n_hard_negatives = cfg.hard_negatives_to_mine
        if cfg.passage_prefix:
            self.passage_prefix = cfg.passage_prefix
        if cfg.query_prefix:
            self.query_prefix = cfg.query_prefix
        if cfg.hard_neg_mining_algorithm:
            self.hard_neg_mining_algorithm = cfg.hard_neg_mining_algorithm
        else:
            print ('hard negative mining algorithm not mentioned in config, using default')
            self.hard_neg_mining_algorithm = "topk_percpos"
        if self.hard_neg_mining_algorithm == "topk_percpos":
            if cfg.percpos:
                self.percpos = cfg.percpos
            else:
                self.percpos = 0.95
        elif self.hard_neg_mining_algorithm == "topk_abs":
            if cfg.max_hardness_threshold:
                self.max_neg_score_threshold = cfg.max_hardness_threshold
            else:
                raise ValueError("Hard negative threshold is required!")
            if cfg.min_hardness_threshold:    
                self.min_neg_score_threshold = cfg.min_hardness_threshold
            else:
                self.min_neg_score_threshold = 0.

        if cfg.min_cluster_size:
            self.min_cluster_size = cfg.min_cluster_size
        if cfg.max_number_clusters:
            self.max_number_clusters = cfg.max_number_clusters
        if cfg.cluster_output_dir:
            self.cluster_output_dir = cfg.cluster_output_dir
        if cfg.logger_output_dir:
            self.logger_output_dir = cfg.logger_output_dir
                
        self._initialize_model()

    def __dask_tokenize__(self):
        return normalize_token(HardNegativeMiner)


    def assign_ids(self, partition):
        return partition.assign(doc_id=np.arange(len(partition)) + partition.index[0])
    

    def repartition_semantic_similarity(self, dataset: DocumentDataset) -> DocumentDataset:
        df = dataset.df
        n_data = df.compute().shape[0] # number of row items
        n_clusters = int(np.floor(n_data/self.min_cluster_size) + 1)
        n_clusters = min(n_clusters, self.max_number_clusters)
        assert "doc_id" not in df.columns
        df['embeddings'] = "" # refers to document embeddings
        df = df.map_partitions(self._get_doc_embeddings, meta = df)
 
        df = df.explode("documents")
        df = df.map_partitions(self.assign_ids)
        embeddings_dataset = DocumentDataset(df)
        
        self.clustering_model = ClusteringModel(
            id_column="doc_id",
            max_iter=100,
            n_clusters= n_clusters,
            clustering_output_dir=self.cluster_output_dir,
            logger=self.logger_output_dir
        )
        clustered_dataset = self.clustering_model(embeddings_dataset)

        return clustered_dataset


    def _get_doc_embeddings(self, p_df: pd.DataFrame):
        if self.model_type == "nvidia":
            p_df['embeddings'] = p_df['documents'].map(lambda pgs: [self._get_nim_embedding(t, "passage") for t in pgs])
        elif self.model_type == "hf":
            p_df['embeddings'] = p_df['documents'].map(lambda pgs: [self._get_hf_embedding(t, self.passage_prefix) for t in pgs])
        return p_df


    def _groupby_question(self, pdf):
        return pdf.groupby("question").agg({'documents':list})

    
    def __call__(self, dataset: DocumentDataset) -> DocumentDataset: 
        
        df = dataset.df
        df = df.to_backend('pandas')
        df = df[['question', 'documents']]
        df = df.map_partitions(self._groupby_question).reset_index()
                                         
        print ("Number partitions in dataset = {}".format(df.npartitions))
        
        df['neg_doc_scores'] = ""
        df['neg_doc'] = ""
        df['doc_embed'] = ""
        df['query_embed'] = ""
        df['min_pos_score'] = ""
        
        with ProgressBar(dt=1):
            df = df.map_partitions(self._process_partition, meta=df).compute()

        df = df.rename(columns={"documents":"pos_doc"})
        df = df[['question','pos_doc','neg_doc']]
        
        return DocumentDataset(dd.from_pandas(df))


    
    def _process_partition(self, df_p: pd.DataFrame):

       
        if self.model_type=="nvidia":
            df_p['doc_embed'] = df_p['documents'].map(lambda pgs: [self._get_nim_embedding(t, "passage") for t in pgs])
            df_p['query_embed'] = df_p['question'].map(lambda x: self._get_nim_embedding(x, "query"))
        elif self.model_type=="hf":
            df_p['doc_embed'] = df_p['documents'].map(lambda pgs: [self._get_hf_embedding(t, self.passage_prefix) for t in pgs])
            df_p['query_embed'] = df_p['question'].map(lambda x: self._get_hf_embedding(x, self.query_prefix))
            
           
        doc_embeds = list(itertools.chain(*df_p['doc_embed'].to_list()))
        docs = list(itertools.chain(*df_p['documents'].to_list()))
      
        
        if self.hard_neg_mining_algorithm == "topk_abs":
            df_p['neg_doc_scores'] = df_p[['query_embed', 
                                           'documents']].apply(lambda row: self._get_scores_topk_abs(row['query_embed'],
                                                                                                     doc_embeds, 
                                                                                                     docs, 
                                                                                                     row['documents']), 
                                                               axis=1)
           
        elif self.hard_neg_mining_algorithm == "topk_percpos":
            df_p['min_pos_score'] = df_p[['query_embed','doc_embed']].apply(lambda row: self._get_min_pos_score(row), axis=1)
            df_p['neg_doc_scores'] = df_p[['query_embed', 
                                           'min_pos_score']].apply(lambda row: self._get_scores_topk_percpos(row,
                                                                                                             doc_embeds, 
                                                                                                             docs), 
                                                                   axis=1)
             
           
        df_p['neg_doc'] = df_p['neg_doc_scores'].map(lambda x: [doc for doc, score in x])
        return df_p
        


    def _get_min_pos_score(self, row):
        x_ = np.array(row['query_embed'])
        y_ = np.array(row['doc_embed']) # document embeddings
        scores  = np.dot(x_, y_.T)
        return np.min(scores)

    

    def _get_scores_topk_percpos(self, row, docs_embed, docs):
        x_ = np.array(row['query_embed'])
        y_ = np.array(docs_embed)
        scores = np.dot(x_, y_.T)
        neg_docs = []
        neg_docs_scores = []
        max_neg_score_threshold = row['min_pos_score'] * self.percpos
        for idx, s in enumerate(scores):
            if s <= max_neg_score_threshold:
                if docs[idx] not in neg_docs:
                    neg_docs.append(docs[idx])
                    neg_docs_scores.append((docs[idx], s))
        del neg_docs, scores            
        return sorted(neg_docs_scores, reverse=True, key=lambda x: x[1])[:self.n_hard_negatives]
        

        
    def _get_scores_topk_abs(self, x, docs_embed, docs, pos_docs):
        x_ = np.array(x)
        y_ = np.array(docs_embed)
        scores = np.dot(x_, y_.T)
        neg_docs = []
        neg_docs_scores = []
        for idx, s in enumerate(scores):
            if s <= self.max_neg_score_threshold:
                if docs[idx] not in pos_docs:
                    if docs[idx] not in neg_docs:
                        neg_docs.append(docs[idx])
                        neg_docs_scores.append((docs[idx], s))
        del neg_docs, scores            
        return sorted(neg_docs_scores, reverse=True, key=lambda x: x[1])[:self.n_hard_negatives]
       
    
    def _initialize_model(self):
        if self.model_type =="nvidia":
            try:
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except Exception as e:
                print(f"Error accessing NIM model: {e}, provide correct API key & URL")
        elif self.model_type =="hf":
            self.hf_model = SentenceTransformer(self.model_name, trust_remote_code=True)
        
        
    def _get_hf_embedding(self, text, prefix="query"):
        embeddings = self.hf_model.encode(prefix + text)
        return embeddings

    
    def _get_nim_embedding(self, text, input_type):
        # Obtain embeddings from nim model
        if isinstance(text, list):
            input_ = text
        elif isinstance(text, str):
            input_ = [text]

        try:
            response = self.client.embeddings.create(
                input=input_,
                model=self.model_name,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": self.truncate},
            )
        except Exception as e:
            print(f"Error: {e}")
            response = None

        if response:
            if isinstance(text, list):
                embeddings = [r.embedding for r in response.data]
            elif isinstance(text, str):
                embeddings = response.data[0].embedding
            return embeddings
        else:
            return []



