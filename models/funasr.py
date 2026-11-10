import numpy as np
from typing import List

from .base import BaseASRModel


class FunASRModel(BaseASRModel):
    """FunASR模型"""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id)
        self.kwargs = kwargs

    def load(self):
        from funasr import AutoModel

        self.model = AutoModel(model=self.model_id, **self.kwargs)

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        res = self.model.generate(input=audio_batch)
        return [item["text"] for item in res]


class paraformer_onnx(BaseASRModel):
    """FunASR ONNX 版 SeacoParaformer"""

    def __init__(self, model_id: str, quantize=True, hotwords: str = ""):
        super().__init__(model_id)
        self.quantize = quantize
        self.hotwords = hotwords
        # "manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
        # "manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline"

    def load(self):
        from funasr_onnx import SeacoParaformer

        self.model = SeacoParaformer(
            self.model_id, batch_size=1, quantize=self.quantize
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        results = [
            "".join(item["preds"] for item in self.model(audio, self.hotwords))
            for audio in audio_batch
        ]
        return results
