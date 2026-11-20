from realtime_speech import start_transcribe
from models.andrewleech.asr_model import Qwen3ASRModel

# modelscope download rhasspy/qwen3-asr-0.6b-onnx-int4-merged --local_dir qwen3-asr-0.6b-onnx-int4-merged

start_transcribe(
    Qwen3ASRModel("models/andrewleech/qwen3-asr-0.6b-onnx-int4-merged"),
    timing=False,
    need_punc=False,
)
