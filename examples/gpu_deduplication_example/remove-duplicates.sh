#! /bin/bash

#SBATCH --job-name=nemo-data-curator:remove-duplicates
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --time=01:00:00

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
log_dir=${src_dir}/workspace/log/remove_duplicates
res_dir=${src_dir}/workspace/data/remove_duplicates
conf_dir=${src_dir}/workspace/config
mkdir -p ${log_dir} ${res_dir} ${conf_dir}

## Container related variables
docker_image="nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11"
mounts="${base_dir}:${base_dir}"

## Set relevant filepaths
input_data_dir="<Specify Paths to dataset>"
input_id_list="<Specify list containing duplicate ids>"
output_data_dir="<Specify output directory to where deduped docs will be written>"
fname=$(basename ${input_id_list})
tag=$(basename $fname .txt)

srun -l \
  --output=${log_dir}/remove_duplicates_${tag}_%j.out \
  --error=${log_dir}/remove_duplicates_${tag}_%j.err \
  --container-image=${docker_image} \
  --container-mounts=${mounts} \
    remove_duplicates \
      --input-data-dir=${input_data_dir} \
      --input-id-list=${input_id_list} \
      --output-deduped-dir=${output_data_dir}/all_deduped \
      --log-dir=${log_dir}/all_deduped_${tag}
