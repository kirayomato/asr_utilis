import numpy as np
from typing import List

from .base import BaseASRModel
from .paths import checkpoint_path

REPO_ID = "owner/model-name"


class MyASRModel(BaseASRModel):
    """<一句话说明这是什么模型的哪条推理路径>

    model_id 是 models/checkpoint 下的目录名，目录缺失时按 REPO_ID 从 ModelScope 下载。
    """

    def __init__(self, model_id: str = "model-name"):
        super().__init__(model_id)

    def load(self):
        model_dir = checkpoint_path(self.model_id, REPO_ID)

        from upstream_lib import Pipeline

        self.model = Pipeline(str(model_dir))

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        return [self.model.transcribe(wav, 16000) for wav in audio_batch]
