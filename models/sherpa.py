import numpy as np
from typing import List

from .base import BaseASRModel


def _decode_streams(
    recognizer, audio_batch: List[np.ndarray], tail_padding: float
) -> List[str]:
    """按流式/非流式两种口径解码一批音频

    online recognizer 需要补一段静音尾部并等解码队列消化完，offline 直接逐条解码。
    """
    import sherpa_onnx

    online = isinstance(recognizer, sherpa_onnx.OnlineRecognizer)
    streams = []
    for wav in audio_batch:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, wav)
        if online:
            paddings = np.zeros(int(tail_padding * 16000), dtype=np.float32)
            stream.accept_waveform(16000, paddings)
            stream.input_finished()
        streams.append(stream)

    if online:
        while True:
            ready_list = [s for s in streams if recognizer.is_ready(s)]
            if len(ready_list) == 0:
                break
            recognizer.decode_streams(ready_list)
        return [recognizer.get_result(s) for s in streams]

    for stream in streams:
        recognizer.decode_stream(stream)
    return [s.result.text for s in streams]


class zipformer(BaseASRModel):
    """sherpa-onnx 里的各类 zipformer / ctc 模型"""

    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self):
        import sherpa_onnx

        if self.model_id == "online_ctc":
            path = "./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "offline_ctc":
            path = "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"

            self.model = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "online_transducer":
            path = "./sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30/"
            tokens = path + "tokens.txt"
            encoder = path + "encoder.int8.onnx"
            decoder = path + "decoder.onnx"
            joiner = path + "joiner.int8.onnx"
            self.model = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                decoding_method="modified_beam_search",
                num_threads=8,
                # hotwords_file=args.hotwords_file,
            )
        elif self.model_id == "dolphin":
            path = "./sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "omnilingual":
            path = (
                "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/"
            )
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "offline_wenet_wenetspeech":
            path = "./sherpa-onnx-zh-wenet-wenetspeech/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "online_wenet_wenetspeech":
            path = "./sherpa-onnx-zh-wenet-wenetspeech/"
            model = path + "model-streaming.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        return _decode_streams(self.model, audio_batch, tail_padding=0.66)


class x_asr(BaseASRModel):
    """X-ASR 流式 zipformer，chunk 以毫秒数区分模型目录"""

    def __init__(self, chunk: str = "1920"):
        super().__init__(f"x_asr_chunk{chunk}")
        self.chunk = chunk

    def load(self):
        import sherpa_onnx

        path = f"X-ASR/deployment/models/chunk-{self.chunk}ms-model/"
        self.model = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=path + "tokens.txt",
            encoder=path + f"encoder-{self.chunk}ms.onnx",
            decoder=path + f"decoder-{self.chunk}ms.onnx",
            joiner=path + f"joiner-{self.chunk}ms.onnx",
            num_threads=8,
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        return _decode_streams(self.model, audio_batch, tail_padding=2.0)
