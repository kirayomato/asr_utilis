import funasr_onnx
import librosa
import numpy as np
import copy
from typing import Any, List, Union
from datetime import datetime, timezone
from os.path import splitext, basename
from os import system, rename, remove
from time import time
from threading import Thread
from colorama import Fore, init, Style
import sherpa_onnx
from tqdm import tqdm
import traceback

# if splitext(video_path)[1] != 'wav':
#     path1 = video_path.replace(' ', '_')
#     rename(video_path, path1)
#     path1 = splitext(path1)[0]+'.wav'
#     system(f'ffmpeg -i {video_path} {path1}')
#     video_path = path1
video = []


def interpolate_shorten_array(arr, target_length):
    """
    使用插值均匀地将长数组缩短到目标长度

    参数:
    arr: 输入数组（列表或numpy数组）
    target_length: 目标长度（整数）

    返回:
    缩短后的数组
    """
    if target_length >= len(arr):
        return arr

    # 创建原始数组的索引
    old_indices = np.linspace(0, len(arr) - 1, len(arr))
    # 创建新数组的索引
    new_indices = np.linspace(0, len(arr) - 1, target_length)

    # 使用线性插值
    shortened_arr = np.interp(new_indices, old_indices, arr)

    return shortened_arr.tolist()


def deal_input():
    while 1:
        video.append(input())


def generate_with_vad(
    self: sherpa_onnx.OnlineRecognizer,
    wav_content: np.ndarray,
    hotwords: str,
    vad: np.ndarray = None,
    **kwargs,
) -> List:
    waveform_list = []
    if vad:
        scale = 16
        for [i, j] in vad:
            waveform_list.append(wav_content[i * scale : j * scale])
    waveform_nums = len(waveform_list)
    asr_res = []

    for idx in tqdm(range(0, waveform_nums)):
        result = zip(self, waveform_list[idx])
        text = result.text.replace("<unk>", "").replace(" ", "")
        ts = result.timestamps
        ts.append(len(waveform_list[idx]) / 16000)
        ts = interpolate_shorten_array(ts, len(text))
        asr_res.append(
            {
                "preds": " ".join(list(text)),
                "timestamp": ts,
            }
        )
    if vad:
        res = {
            "preds": "",
            "timestamp": [],
        }
        for idx in range(len(asr_res)):
            m = len(asr_res[idx]["timestamp"])
            for i in range(m):
                asr_res[idx]["timestamp"][i] += vad[idx][0] / 1000
            res["preds"] += asr_res[idx]["preds"] + " "
            res["timestamp"] += asr_res[idx]["timestamp"]
        return res
    return asr_res


def add_newlines_after_punctuation(text, punctuation_marks="。.,，!！?？;:"):
    result = []
    for char in text:
        result.append(char)
        if char in punctuation_marks:
            result.append("\n")
    return "".join(result)


def generate(video_path):
    t0 = time()
    audio = librosa.load(video_path, sr=16000)[0]

    ouput_path = splitext(video_path)[0] + f"_{model_name}.srt"
    raw = splitext(video_path)[0] + f"_{model_name}.txt"

    # t2 = time()
    vad_result = vad(audio)[0]
    # print(f"VAD:{time()-t2:.2f}s")

    res = generate_with_vad(
        model, audio, hotwords="雨纪Ameki 雨纪 ameki 维阿 罐罐", vad=vad_result
    )
    text = res["preds"].split()
    ts = res["timestamp"]
    # print(f"ASR:{time()-t0:.2f}s")
    # t1 = time()
    assert len(text) == len(ts), "时间戳和文本长度不一致"
    res = punc(res["preds"])[0]
    # print(f"PUNC:{time()-t1:.2f}s")

    with open(raw, "w", encoding="utf-8") as f1:
        f1.write(add_newlines_after_punctuation(res))

    with open(ouput_path, "w", encoding="utf-8") as f:
        n = len(ts)
        m = len(res)
        cnt = 1
        s = ""
        i = 0
        j = 0
        while i < n:
            f.write(f"{cnt:d}\n")
            dt = datetime.fromtimestamp(ts[i], timezone.utc)
            f.write(dt.strftime("%H:%M:%S,%f") + " --> ")
            while res[j : j + len(text[i])].lower() == text[i].lower():
                if len(s) > 12:
                    j -= 1
                    break
                s += text[i]
                j += len(text[i])
                i += 1
                if i >= n or j + len(text[i]) > m:
                    break
            if s == "":
                pass
            dt = datetime.fromtimestamp(ts[i - 1], timezone.utc)
            f.write(dt.strftime("%H:%M:%S,%f") + "\n")
            f.write(s + "\n\n")
            s = ""
            j += 1
            cnt += 1

    print(f"Convert:{time()-t0:.2f}s")
    system(f'"{ouput_path}"')


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
    elif model == "omnilingual":
        path = "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/"
        model = path + "model.int8.onnx"
        tokens = path + "tokens.txt"
        return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
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
    elif model == "offline_wenet_multi_cn":
        path = "./sherpa-onnx-zh-wenet-multi-cn/"
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
    elif model == "online_wenet_multi_cn":
        path = "./sherpa-onnx-zh-wenet-multi-cn/"
        model = path + "model-streaming.int8.onnx"
        tokens = path + "tokens.txt"
        return sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
            model=model,
            tokens=tokens,
            num_threads=8,
        )


def zip(model, wav):
    stream = model.create_stream()
    stream.accept_waveform(16000, wav)
    tail_paddings = np.zeros(int(0.66 * 16000), dtype=np.float32)
    stream.accept_waveform(16000, tail_paddings)
    stream.input_finished()
    if isinstance(model, sherpa_onnx.OnlineRecognizer):
        while model.is_ready(stream):
            model.decode_stream(stream)
        return model.get_result_all(stream)
    else:
        model.decode_stream(stream)
        return stream.result


if __name__ == "__main__":
    init()
    t1 = Thread(target=deal_input)
    t1.start()
    t0 = time()
    from funasr_onnx import CT_Transformer, Fsmn_vad

    model_name = "offline_wenet_multi_cn"
    model = create_recognizer(model_name)
    model_vad = "iic/speech_fsmn_vad_zh-cn-16k-common-onnx"
    vad = Fsmn_vad(model_vad, batch_size=1, quantize=True)
    model_ct = "iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx"
    punc = CT_Transformer(model_ct, batch_size=1, quantize=True)
    print(f"Load:{time()-t0:.2f}s")
    while 1:
        if len(video):
            video_path = video.pop()
            try:
                generate(video_path)
            except BaseException as e:
                print(
                    Fore.RED + f"ERROR:{e}\n{traceback.format_exc()}" + Style.RESET_ALL
                )
