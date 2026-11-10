import numpy as np
from typing import List

from .base import BaseASRModel


class Nemotron(BaseASRModel):
    """NVIDIA Nemotron RNNT"""

    def __init__(self, model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"):
        super().__init__(model_id)

    def load(self):
        from transformers import AutoModelForRNNT, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForRNNT.from_pretrained(self.model_id, device_map="auto")

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        result = []
        for wav in audio_batch:
            inputs = self.processor(
                wav,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                language="zh-CN",
            )
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            text, ts = self.processor.decode(
                output.sequences, durations=output.durations, skip_special_tokens=True
            )
            result.append(text[0])
        return result
