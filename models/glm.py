import numpy as np
import torch
from typing import List

from .base import BaseASRModel


class GLMASRLLMModel(BaseASRModel):
    """GLM-ASR LLM 版"""

    def __init__(self, model_id: str = "GLM-ASR-Nano-2512", **kwargs):
        super().__init__(model_id)
        self.kwargs = kwargs

    def load(self):
        from transformers import AutoModel, AutoProcessor

        repo_id = f"zai-org/{self.model_id}"
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = AutoModel.from_pretrained(
            repo_id, dtype=torch.bfloat16, device_map="auto", **self.kwargs
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        inputs = self.processor.apply_transcription_request(audio_batch)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=500, pad_token_id=59246
        )
        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded_outputs
