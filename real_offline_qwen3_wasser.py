from realtime_speech import start_transcribe
from models.wasser1462.asr_model import Qwen3ASRModel

# modelscope download jkman2023/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 --local_dir sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25

start_transcribe(
    Qwen3ASRModel(
        model_id="models/wasser1462/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        model_dir="models/wasser1462/Qwen3-ASR-0.6B-meta",
    ),
    timing=False,
    need_punc=False,
)
