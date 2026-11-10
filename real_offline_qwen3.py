import logging
import os

from asr_eval import Qwen3ASRModel
from realtime_speech import start_transcribe

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

start_transcribe(
    Qwen3ASRModel("Qwen3-ASR-0.6B", language=None, pad_token_id=151645),
    need_punc=False,
    timing=False,
)
