import numpy as np
import torch
from typing import List

from .base import BaseASRModel


class WhisperModel(BaseASRModel):
    """Whisper模型"""

    def __init__(self, model_id: str, device: str = None):
        super().__init__(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        import whisper

        self.model = whisper.load_model(self.model_id, device=self.device)

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        results = []
        for audio in audio_batch:
            audio_tensor = torch.from_numpy(audio).float()
            result = self.model.transcribe(audio_tensor)
            results.append(result["text"])
        return results
