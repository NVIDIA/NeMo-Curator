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

import cudf
import numpy as np


def compute_jaccard_partition(df):
    df["jaccard"] = df["text_x"].str.jaccard_index(df["text_y"], width=5)
    df.drop(columns=["text_x", "text_y"], inplace=True)
    return df


def get_max_num_rows_to_process_once(df):
    nbytes = df["text"].str.byte_count().sum()
    # Number of exmploded bytes
    exploded_bytes = nbytes * 5 * 2
    max_chars_allowed = 2_147_483_647
    byte_ratio = int(exploded_bytes) // max_chars_allowed
    if byte_ratio > 1:
        nrows_at_once = len(df) // byte_ratio
    else:
        nrows_at_once = len(df)

    nrows_at_once = max(1, nrows_at_once)
    return nrows_at_once


def create_empty_jaccard_result():
    df = cudf.DataFrame()
    df["adlr_id_x"] = "x"
    df["adlr_id_y"] = "y"
    df["jaccard"] = np.empty(shape=0, dtype=np.float32)
    return df


def compute_jaccard_pair(docs_df, anchor_df):
    nrows_at_once = get_max_num_rows_to_process_once(docs_df)
    result_ls = []
    for i in range(0, docs_df.shape[0], nrows_at_once):
        pair_df = docs_df[i : i + nrows_at_once]
        pair_df = pair_df.merge(anchor_df, on="anchor_adlr_id")
        pair_df = pair_df.rename(
            columns={"adlr_id": "adlr_id_x", "anchor_adlr_id": "adlr_id_y"}
        )
        mask = pair_df.adlr_id_x != pair_df.adlr_id_y
        pair_df = pair_df[mask].reset_index(drop=True)
        if len(pair_df) == 0:
            result_df = create_empty_jaccard_result()
        else:
            result_df = compute_jaccard_partition(pair_df)
        result_ls.append(result_df)
    if len(result_ls) == 0:
        return create_empty_jaccard_result()
    df_pair = cudf.concat(result_ls)
    return df_pair


def get_anchor_df(df, anchor_col):
    anchor_df = df[df["adlr_id"] == df[anchor_col]]
    anchor_df = anchor_df.reset_index(drop=True)
    anchor_df = anchor_df[[anchor_col, "text"]]
    anchor_df = anchor_df.rename(columns={anchor_col: "anchor_adlr_id"})
    return anchor_df


def compute_jaccard_and_create_pair_df(df):
    df = df.drop_duplicates(
        subset=["adlr_id", "anchor_1_adlr_id", "anchor_0_adlr_id"], ignore_index=True
    )
    anchor_columns = ["anchor_0_adlr_id", "anchor_1_adlr_id"]
    result_ls = []
    try:
        for anchor_col in anchor_columns:
            doc_df = df[["adlr_id", "text", anchor_col]]
            doc_df = doc_df.rename(columns={anchor_col: "anchor_adlr_id"})
            doc_df = doc_df[doc_df["adlr_id"] != doc_df["anchor_adlr_id"]]
            anchor_df = get_anchor_df(df, anchor_col)
            result_df = compute_jaccard_pair(doc_df, anchor_df)
            result_ls.append(result_df)

        return cudf.concat(result_ls)
    except OverflowError as e:
        print(
            "Failed with  OverflowError in compute_jaccard_and_create_pair_df",
            flush=True,
        )
        print(df, flush=True)
        print("--" * 30)
        print("Error")
        print("---" * 30)
        raise e
