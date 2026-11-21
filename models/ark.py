import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from .base import BaseASRModel
from .paths import checkpoint_path

WAV_PATH = Path(tempfile.gettempdir()) / "ark_asr_soundfile.wav"
REPO_ID = "Audio8/ark-asr-0.6b-int8-onnx"


class ark_audio_onnx(BaseASRModel):
    """方舟 ONNX 版 ASR"""

    def __init__(self, model_id: str = "ark-asr-0.6b-int8-onnx"):
        super().__init__(model_id)

    def load(self):
        model_dir = checkpoint_path(self.model_id, REPO_ID).resolve()

        self.model = _build_runtime(model_dir)

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        result = []
        for wav in audio_batch:
            sf.write(WAV_PATH, wav, 16000, subtype="PCM_16")
            text = self.model.transcribe(
                audio_path=str(WAV_PATH),
                max_new_tokens=128,
                max_audio_seconds=30,
                precision="int8",
                asr_block_token_id_from=151670,
            )
            result.append(text)
        return result


def _build_runtime(package_dir: Path):
    sys.path.insert(0, str(package_dir))
    from infer_ark_audio_onnx import ArkAsrOnnxRuntime

    class FlatLayoutRuntime(ArkAsrOnnxRuntime):
        """上游按 <root>/model + <root>/build 找文件，下载包是扁平结构，这里让两个目录都指向包根"""

        def __init__(self, package_dir: Path):
            self._package_dir = package_dir
            super().__init__(package_dir)

        @property
        def model_dir(self) -> Path:
            return self._package_dir

        @model_dir.setter
        def model_dir(self, value):
            pass

        @property
        def build_dir(self) -> Path:
            return self._package_dir

        @build_dir.setter
        def build_dir(self, value):
            pass

    return FlatLayoutRuntime(package_dir)
