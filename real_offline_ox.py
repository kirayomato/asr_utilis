from models import paraformer_onnx
from realtime_speech import start_transcribe

start_transcribe(
    paraformer_onnx(
        "manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline",
        hotwords="雨纪 ameki 米汀 米米 维阿 罐罐 百舰 舰长 点酥 酥酥 海海 栞栞 岁己",
    ),
    timing=False,
)
