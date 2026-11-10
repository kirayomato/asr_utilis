from asr_eval import GLMASRLocalModel
from realtime_speech import start_transcribe

start_transcribe(
    GLMASRLocalModel(r"D:\Program\VLM\cache\ZhipuAI\GLM-ASR-Nano-2512"),
    need_punc=False,
)
