import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List

from .base import BaseASRModel


class ark_audio_onnx(BaseASRModel):
    """方舟 ONNX 版 ASR"""

    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self):
        os.chdir("ark-asr-0.6b-int8-onnx")
        from infer_ark_audio_onnx import ArkAsrOnnxRuntime

        self.model = ArkAsrOnnxRuntime(Path("."))

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        result = []
        for wav in audio_batch:
            sf.write("soundfile.wav", wav, 16000, subtype="PCM_16")
            text = self.model.transcribe(
                audio_path="soundfile.wav",
                max_new_tokens=128,
                max_audio_seconds=30,
                precision="int8",
                asr_block_token_id_from=151670,
            )
            result.append(text)
        return result
