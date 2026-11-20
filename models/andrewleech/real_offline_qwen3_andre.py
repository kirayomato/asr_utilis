from realtime_speech import start_transcribe
from asr_model import Qwen3ASRModel

# modelscope download rhasspy/qwen3-asr-0.6b-onnx-int4-merged --local_dir qwen3-asr-0.6b-onnx-int4-merged

start_transcribe(
    Qwen3ASRModel(
        "qwen3-asr-0.6b-onnx-int4-merged",
        context="词汇: 雨纪,ameki,米汀,米米,维阿,罐罐,百舰,舰长,点酥,酥酥,海海,栞栞,岁己",
        language="Chinese",
    ),
    timing=False,
)
