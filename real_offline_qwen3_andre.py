from models import checkpoint_path
from models.andrewleech.asr_model import Qwen3ASRModel
from realtime_speech import start_transcribe

REPO_ID = "rhasspy/qwen3-asr-0.6b-onnx-int4-merged"

start_transcribe(
    Qwen3ASRModel(str(checkpoint_path("qwen3-asr-0.6b-onnx-int4-merged", REPO_ID))),
    timing=False,
    need_punc=False,
)
