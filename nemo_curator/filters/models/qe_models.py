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

from abc import ABC, abstractmethod
from typing import List

try:
    from comet import download_model, load_from_checkpoint
    use_comet = True
except ImportError:
    use_comet = False


class QEModel(ABC):

    def __init__(self, model, gpu=False):
        self._model = model
        self._gpu = gpu

    @classmethod
    @abstractmethod
    def load_model(cls, model_name: str):
        pass

    @staticmethod
    @abstractmethod
    def wrap_qe_input(src: str, tgt: str, reverse=False, **kwargs):
        pass

    @abstractmethod
    def predict(self, src: str, tgt: str, **kwargs) -> List[float]:
        pass


class COMETQEModel(QEModel):

    MODEL_NAME_TO_HF_PATH = {
        "comet-qe": "Unbabel/wmt20-comet-qe-da",
    }

    @classmethod
    def load_model(cls, model_name: str, gpu: bool = False):
        if not use_comet:
            raise RuntimeError(
                'To run QE filtering with COMET, you need to install from PyPI with: `pip install unbabel-comet`. '
                'More information at https://github.com/Unbabel/COMET.'
            )

        path = download_model(cls.MODEL_NAME_TO_HF_PATH[model_name])
        return cls(load_from_checkpoint(path), gpu)

    @staticmethod
    def wrap_qe_input(src: str, tgt: str, reverse=False, **kwargs):
        return {"src": src, "mt": tgt} if not reverse else {"src": tgt, "mt": src}

    def predict(self, input: List, **kwargs) -> List[float]:
        return self._model.predict(input, gpus=int(self._gpu), num_workers=0).scores  # it's critical to set num_workers=0 to avoid spawning new processes within a dask worker
