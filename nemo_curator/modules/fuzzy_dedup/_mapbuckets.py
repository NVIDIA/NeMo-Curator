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

from __future__ import annotations

import logging
import os
from typing import Union

import cudf
import dask_cudf
import numpy as np
from dask.utils import M

from nemo_curator.log import create_logger


class _MapBuckets:
    """
    buckets to a logical partition by using a modified bin packing algorithm.
    Combines buckets generated from LSH (typically high cardinality)
    to more coarse lower cardinality bucket groups by mapping multiple buckets
    to a logical partition using document length information and a modified bin
    packing algorithm.
    Only needed if running False Postive check to remove false positives.
    """

    def __init__(
        self,
        id_fields: Union[list, str] = "id",
        text_field: str = "text",
        bucket_field: str = "_bucket_id",
        num_anchors: int = 2,
        logger: Union[logging.LoggerAdapter, str] = "./",
    ):
        """
        id_fields: list or str
            id fields of df
        text_field: str = "text",
        bucket_column: str = "bucket_column",
        num_anchors: int = 2,
        logger: Union[logging.LoggerAdapter, str] = "./",
        """
        self.id_fields = [id_fields] if isinstance(id_fields, str) else id_fields
        self.text_field = text_field
        self.num_anchors = num_anchors
        self.bucket_field = bucket_field
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "Map_Buckets.log"),
                name="Map_Buckets",
            )
        else:
            self._logger = logger

    def _random_select_anchor(self, buckets_df, n=2):
        """
        Randomly select `n` anchors from each bucket.
        """
        buckets_df = buckets_df.copy()
        buckets_df["_id_hash"] = buckets_df[self.id_fields].hash_values()
        buckets_df = buckets_df.sort_values([self.bucket_field, "_id_hash"])
        buckets_df["_order_in_bucket"] = buckets_df.groupby(
            self.bucket_field
        ).cumcount()
        buckets_df["is_anchor"] = buckets_df["_order_in_bucket"] < n
        for i in range(0, n):
            buckets_df[f"is_anchor_id_{i}"] = buckets_df["_order_in_bucket"] == i
        buckets_df = buckets_df.drop(columns=["_id_hash", "_order_in_bucket"], axis=1)
        buckets_df = buckets_df.reset_index(drop=True)
        buckets_df = buckets_df[buckets_df.is_anchor]
        return buckets_df

    def _add_anchor_docs(self, buckets_df, num_anchors):
        """
        Get anchor documents for each bucket.
        """
        df_anchor_bk = self._random_select_anchor(buckets_df=buckets_df, n=num_anchors)
        df_anchor_docs = None
        for i in range(num_anchors):
            df_anchor_bk_i = df_anchor_bk[df_anchor_bk[f"is_anchor_id_{i}"]][
                [self.bucket_field] + self.id_fields
            ].reset_index(drop=True)
            column_mapping = {id: f"anchor_{i}_{id}" for id in self.id_fields}
            df_anchor_bk_i = df_anchor_bk_i.rename(columns=column_mapping)
            if i == 0:
                df_anchor_docs = df_anchor_bk_i
            else:
                df_anchor_docs = df_anchor_bk_i.merge(
                    df_anchor_docs, on=[self.bucket_field], how="inner"
                )

        df_anchor_docs_with_bk = buckets_df.merge(
            df_anchor_docs, on=[self.bucket_field], how="inner"
        )
        return df_anchor_docs_with_bk

    def map_buckets_with_anchors(
        self,
        buckets_df: dask_cudf.DataFrame,
        shuffle_type: Union[str, bool, None] = "tasks",
    ) -> dask_cudf.DataFrame:
        ddf_anchor_docs_with_bk = buckets_df.map_partitions(
            self._add_anchor_docs, num_anchors=self.num_anchors
        )

        # Bucket is no longer needed
        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.drop(
            columns=[self.bucket_field]
        )

        # Below removes any duplicates lying around after dropping buckets
        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.map_partitions(
            M.drop_duplicates,
            meta=ddf_anchor_docs_with_bk._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )

        ddf_anchor_docs_with_bk = ddf_anchor_docs_with_bk.shuffle(
            self.id_fields,
            ignore_index=True,
            shuffle_method=shuffle_type,
        ).map_partitions(
            M.drop_duplicates,
            meta=ddf_anchor_docs_with_bk._meta,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
        )

        return ddf_anchor_docs_with_bk
