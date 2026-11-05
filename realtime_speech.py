from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Callable
import pyaudio
from time import localtime, strftime, time
import numpy as np
from func_timeout import func_timeout
import unicodedata


def is_punctuation(char):
    return unicodedata.category(char).startswith("P")


waiting = Queue()


def set_device():
    try:
        return func_timeout(5, lambda: input("device:").lower() or "1")
    except:
        return "1"


def merge_strings(a, b):
    max_length = min(len(a), len(b))
    overlap_length = 0

    # 找到最长的重叠部分
    for i in range(1, max_length + 1):
        if a[-i:] == b[:i]:
            overlap_length = i

    # 合并字符串
    return b[overlap_length:]


def get_current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


def record():
    dev = set_device()
    FORMAT = pyaudio.paInt16  # 音频格式
    CHANNELS = 1  # 通道数（立体声）
    RATE = 16000  # 采样率
    CHUNK = 1000  # 缓冲区大小
    RECORD_SECONDS = 5  # 每次录音的时长
    OVERLAP = 0.3
    WAV_OUTPUT_FILENAME = "output_"  # 输出文件名前缀
    n = int(RATE * RECORD_SECONDS / CHUNK)
    # 初始化PyAudio
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    device = 1
    for i in range(numdevices):
        t = audio.get_device_info_by_host_api_device_index(0, i)
        if (
            t.get("maxInputChannels", 0) > 0
            and t["name"] == f"Line {dev} (Virtual Audio Cable)"
        ):
            device = t["index"]
            break
    # 打开音频流
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device,
        frames_per_buffer=CHUNK,
    )
    print("开始录音...")
    frames = []
    while True:
        # 录制音频
        k = int((OVERLAP / RECORD_SECONDS) * min(len(frames), n))
        frames = frames[-k:]
        early_stop = False
        if k == 0:
            frames = []
        cnt = 0
        for _ in range(n):
            data = stream.read(CHUNK)
            if np.frombuffer(data, np.short).max() < 10:
                cnt += 1
                if cnt > 0.3 * RATE / CHUNK:
                    early_stop = True
                    break
                continue
            else:
                cnt = 0
                frames.append(data)

        if len(frames) > k:
            # 保存WAV文件
            # timestamp = int(time())
            # wav_filename = f"{WAV_OUTPUT_FILENAME}{timestamp}.wav"
            # wf = wave.open(wav_filename, 'wb')
            # wf.setnchannels(CHANNELS)
            # wf.setsampwidth(audio.get_sample_size(FORMAT))
            # wf.setframerate(RATE)
            # wf.writeframes(b''.join(frames))
            # wf.close()

            # # print(f"保存为: {wav_filename}")
            # waiting.put(wav_filename)
            waveform = (
                np.frombuffer(b"".join(frames), dtype=np.short).astype(np.float32)
                / 32768.0
            )
            waiting.put((waveform, early_stop))


def generate(model: Callable[[np.ndarray], str], need_punc=True, timing=False):
    if need_punc:
        from funasr_onnx import CT_Transformer

        model_ct = "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx"
        punc = CT_Transformer(model_ct, batch_size=1, quantize=True)

    dt = datetime.now()
    filename = f"transcribe/transcribe_{dt.strftime('%Y-%m-%d_%H%M%S')}.txt"
    last_s = ""
    cnt = 0
    time0 = time()
    print("\n" + get_current_time() + ": ", flush=True)
    while True:
        wav_file, early_stop = waiting.get()
        try:
            t0 = time()
            res = model(wav_file)
            if need_punc and res:
                try:
                    res = punc(res)[0]
                except:
                    pass
            else:
                res = res.replace(" ", "")
            if timing:
                print(f"\n{time()-t0:.2f}s")
            if len(res):
                if early_stop and not is_punctuation(res[-1]):
                    res += "，"
                s = merge_strings(last_s, res[:-1])
                last_s += s
                cnt += len(s)
                last_s = last_s[-25:]
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(s)
                    print(s, end="")
                    if time() - time0 > 300:
                        print("\n" + get_current_time() + ": ", flush=True)
                        time0 = time()
                    if cnt > 25:
                        f.write("\n")
                        print()
                        cnt = 0
        except BaseException as e:
            print(e)


def start_transcribe(
    transcribe: Callable[[np.ndarray], str], need_punc=True, timing=False
):
    t1 = Thread(target=record)
    t2 = Thread(target=generate, args=[transcribe, need_punc, timing])

    t1.start()
    t2.start()
    t1.join()
    t2.join()
