#! /bin/bash

#SBATCH --job-name=nemo-data-curator:gpu-deduplication
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --time=04:00:00

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

#
# This script can be used for running both exact and fuzzy document-level
# deduplication using Dask and cuDF
#

base_dir=`pwd` # Assumes base dir is top-level dir of repo
RUNSCRIPT=${RUNSCRIPT:-${base_dir}/examples/gpu_deduplication_example/run-minhash.sh}
LIBCUDF_CUFILE_POLICY=${LIBCUDF_CUFILE_POLICY:-OFF}
echo $RUNSCRIPT

docker_image='nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.08.03'
mounts="${base_dir}:${base_dir}"

srun -l \
  --container-mounts=${mounts} \
  --container-image=${docker_image} \
    bash -c "echo ${RUNSCRIPT};echo ${LIBCUDF_CUFILE_POLICY}; LIBCUDF_CUFILE_POLICY=${LIBCUDF_CUFILE_POLICY} RUNSCRIPT=${RUNSCRIPT} bash ${base_dir}/examples/gpu_deduplication_example/run-workflow.sh"
