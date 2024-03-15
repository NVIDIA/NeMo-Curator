#! /bin/bash

#SBATCH --job-name=nemo-data-curator:create-fuzzy-dup-id-list
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0:30:00

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

set -eux

## Log and intermediate results dirs
base_dir=`pwd`
src_dir="${base_dir}/workspace/nemo-data-curator"
log_dir=${src_dir}/workspace/log/create_fuzzy_dup_id_list
res_dir=${src_dir}/workspace/data/create_fuzzy_dup_id_list
conf_dir=${src_dir}/workspace/config
mkdir -p ${log_dir} ${res_dir} ${conf_dir}

## Container related variables
docker_image="nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11"
mounts="${base_dir}:${base_dir}"

## Set relevant filepath
input_id_list_dir=<Provide path to connected_components.parquet generated from fuzzy dedup>

# Generate the mapping and prepare the connected components
srun -l \
  --nodes=1 \
  --output=${log_dir}/create_fuzzy_dup_id_list_%j.out \
  --error=${log_dir}/create_fuzzy_dup_id_list_%j.err \
  --container-image=${docker_image} \
  --container-mounts=${mounts} \
    prepare_fuzzy_ids \
      --path-to-connected-components=${input_id_list_dir} \
      --output-indexed-connected-components=${res_dir}/indexed_connected_components.parquet \
      --output-id-mapping=${res_dir}/mapping.json

srun -l \
  --mpi=pmix \
  --output=${log_dir}/create_fuzzy_dup_id_list_%j.out \
  --error=${log_dir}/create_fuzzy_dup_id_list_%j.err \
  --container-image=${docker_image} \
  --container-mounts=${mounts} \
    create_list_of_duplicate_ids \
      --input-id-list-dir=${res_dir}/indexed_connected_components.parquet \
      --input-bucket-key="group" \
      --id-mapping=${res_dir}/mapping.json \
      --output-id-list-dir=${res_dir}/fuzzy_dup_ids \
      --output-bucket-list-dir=${res_dir}/buckets \
      --log-dir=${log_dir}/create_fuzzy_dup_id_list

# Concatenate the extracted list of ids
cat ${res_dir}/fuzzy_dup_ids/*.txt > ${res_dir}/fuzzy_duplicate_id_list.txt
