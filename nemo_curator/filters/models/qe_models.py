from abc import ABC, abstractmethod
from comet import download_model, load_from_checkpoint
from typing import List


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
        path = download_model(cls.MODEL_NAME_TO_HF_PATH[model_name])
        return cls(load_from_checkpoint(path), gpu)

    @staticmethod
    def wrap_qe_input(src: str, tgt: str, reverse=False, **kwargs):
        return {"src": src, "mt": tgt} if not reverse else {"src": tgt, "mt": src}

    def predict(self, input: List, **kwargs) -> List[float]:
        return self._model.predict(input, gpus=int(self._gpu), num_workers=0).scores
