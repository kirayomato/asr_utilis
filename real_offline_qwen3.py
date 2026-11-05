import logging
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
from realtime_speech import start_transcribe
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
model = Qwen3ASRModel.from_pretrained(
    # r"F:\Program\cache\modelscope\models\Qwen\Qwen3-ASR-0___6B",
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32,  # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256,  # Maximum number of tokens to generate. Set a larger value for long audio input.
    pad_token_id=151645,  # Explicitly set pad_token_id to avoid warning
)


def transcribe(wav: np.ndarray) -> str:
    res = model.transcribe(audio=(wav, 16000))[0]
    return res.text


start_transcribe(transcribe, False, False)
