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

from dataclasses import dataclass
from typing import Dict

import nemo_sdk as sdk


@dataclass
class SlurmJobConfig:
    job_dir: str
    container_entrypoint: str
    script_command: str
    device: str = "cpu"
    interface: str = "eth0"
    protocol: str = "tcp"
    cpu_worker_memory_limit: str = "0"
    rapids_no_initialize: str = "1"
    cudf_spill: str = "1"
    rmm_scheduler_pool_size: str = "1GB"
    rmm_worker_pool_size: str = "72GiB"
    libcudf_cufile_policy: str = "OFF"

    def to_script(
        self, add_scheduler_file: bool = True, add_device: bool = True
    ) -> sdk.Script:
        """
        Converts to a script object executable by NeMo SDK
        Args:
            add_scheduler_file: Automatically appends a '--scheduler-file' argument to the
                script_command where the value is job_dir/logs/scheduler.json. All
                scripts included in NeMo Curator accept and require this argument to scale
                properly on SLURM clusters.
            add_device: Automatically appends a '--device' argument to the script_command
                where the value is the member variable of device. All scripts included in
                NeMo Curator accept and require this argument.
        Returns:
            A NeMo SDK Script that will intialize a Dask cluster, and run the specified command.
            It is designed to be executed on a SLURM cluster
        """
        env_vars = self._build_env_vars()

        if add_scheduler_file:
            env_vars[
                "SCRIPT_COMMAND"
            ] += f" --scheduler-file={env_vars['SCHEDULER_FILE']}"
        if add_device:
            env_vars["SCRIPT_COMMAND"] += f" --device={env_vars['DEVICE']}"

        return sdk.Script(path=self.container_entrypoint, env=env_vars)

    def _build_env_vars(self) -> Dict[str, str]:
        env_vars = vars(self)
        # Convert to uppercase to match container_entrypoint.sh
        env_vars = {key.upper(): val for key, val in env_vars.items()}

        env_vars["LOGDIR"] = f"{self.job_dir}/logs"
        env_vars["PROFILESDIR"] = f"{self.job_dir}/profiles"
        env_vars["SCHEDULER_FILE"] = f"{env_vars['LOGDIR']}/scheduler.json"
        env_vars["SCHEDULER_LOG"] = f"{env_vars['LOGDIR']}/scheduler.log"
        env_vars["DONE_MARKER"] = f"{env_vars['LOGDIR']}/done.txt"

        return env_vars
