from models import ark_audio_onnx
from realtime_speech import start_transcribe

start_transcribe(
    ark_audio_onnx("ark-asr-0.6b-int8-onnx"),
    need_punc=False,
    timing=False,
)
