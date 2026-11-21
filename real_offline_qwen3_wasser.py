from pathlib import Path

from models import checkpoint_path
from models.wasser1462.asr_model import Qwen3ASRModel
from realtime_speech import start_transcribe

REPO_ID = "jkman2023/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"

# tokenizer/mel 元数据不是模型，跟随 wasser1462 包存放
META_DIR = Path(__file__).resolve().parent / "models" / "wasser1462" / "Qwen3-ASR-0.6B-meta"

start_transcribe(
    Qwen3ASRModel(
        model_id=str(checkpoint_path("sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25", REPO_ID)),
        model_dir=str(META_DIR),
    ),
    timing=False,
    need_punc=False,
)
