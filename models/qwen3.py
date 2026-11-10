import numpy as np
import torch
from typing import List, Optional

from .base import BaseASRModel


class Qwen3ASRModel(BaseASRModel):
    """Qwen3ASR模型

    language 为 None 时不下发该参数，语种交由模型自行判断。
    """

    def __init__(self, model_id: str, language: Optional[str] = "Chinese", **kwargs):
        super().__init__(model_id)
        self.language = language
        self.kwargs = kwargs

    def load(self):
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            f"Qwen/{self.model_id}",
            # r"F:\Program\cache\modelscope\models\Qwen\Qwen3-ASR-0___6B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32,  # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256,  # Maximum number of tokens to generate. Set a larger value for long audio input.
            **self.kwargs,
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        audio = [(i, 16000) for i in audio_batch]
        lang = {} if self.language is None else {"language": self.language}
        res = self.model.transcribe(audio=audio, **lang)
        return [item.text for item in res]
