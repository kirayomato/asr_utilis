import numpy as np
import torch
from typing import List

from .base import BaseASRModel


class Wav2Vec2Model(BaseASRModel):
    """Wav2Vec2模型"""

    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        # 预处理音频
        inputs = self.processor(
            audio_batch, sampling_rate=16000, return_tensors="pt", padding=True
        )

        # 移动到GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # 解码
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)

        return transcriptions
