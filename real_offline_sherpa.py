import numpy as np
import sherpa_onnx
from realtime_speech import start_transcribe


def create_recognizer(model):
    if model == "online_ctc":
        path = "./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/"
        model = path + "model.int8.onnx"
        tokens = path + "tokens.txt"
        return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            model=model,
            tokens=tokens,
            num_threads=8,
        )
    elif model == "offline_ctc":
        path = "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/"
        model = path + "model.int8.onnx"
        tokens = path + "tokens.txt"

        return sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=model,
            tokens=tokens,
            num_threads=8,
        )
    elif model == "online_transducer":
        path = "./sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30/"
        tokens = path + "tokens.txt"
        encoder = path + "encoder.int8.onnx"
        decoder = path + "decoder.onnx"
        joiner = path + "joiner.int8.onnx"
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            decoding_method="modified_beam_search",
            num_threads=8,
            # hotwords_file=args.hotwords_file,
        )
    elif model == "dolphin":
        path = "./sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/"
        model = path + "model.int8.onnx"
        tokens = path + "tokens.txt"
        return sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
            model=model,
            tokens=tokens,
            num_threads=8,
        )
    elif model == "offline_wenet_wenetspeech":
        path = "./sherpa-onnx-zh-wenet-wenetspeech/"
        model = path + "model.int8.onnx"
        tokens = path + "tokens.txt"
        return sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
            model=model,
            tokens=tokens,
            num_threads=8,
        )
    elif model == "online_wenet_wenetspeech":
        path = "./sherpa-onnx-zh-wenet-wenetspeech/"
        model = path + "model-streaming.int8.onnx"
        tokens = path + "tokens.txt"
        return sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
            model=model,
            tokens=tokens,
            num_threads=8,
        )


def x_asr(chunk):
    path = f"X-ASR/deployment/models/chunk-{chunk}ms-model/"
    tokens = path + "tokens.txt"
    encoder = path + f"encoder-{chunk}ms.onnx"
    decoder = path + f"decoder-{chunk}ms.onnx"
    joiner = path + f"joiner-{chunk}ms.onnx"
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=8,
        # hotwords_file=args.hotwords_file,
    )


model = x_asr("1920")


def transcribe(wav: np.ndarray) -> str:
    stream = model.create_stream()
    stream.accept_waveform(16000, wav)
    if isinstance(model, sherpa_onnx.OnlineRecognizer):
        tail_paddings = np.zeros(int(2 * 16000), dtype=np.float32)
        stream.accept_waveform(16000, tail_paddings)
        stream.input_finished()
        while model.is_ready(stream):
            model.decode_stream(stream)
        return model.get_result(stream)
    else:
        model.decode_stream(stream)
        return stream.result.text


start_transcribe(transcribe, need_punc=False, timing=True)
