from datetime import datetime
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
from queue import Queue
from threading import Thread
import numpy as np
from func_timeout import func_timeout

import pyaudio
import wave
import time

waiting = Queue()


def set_language():
    try:
        return func_timeout(5, lambda: input("language:").lower())
    except:
        return "zh"


def merge_strings(a, b):
    max_length = min(len(a), len(b))
    overlap_length = 0

    # 找到最长的重叠部分
    for i in range(1, max_length + 1):
        if a[-i:] == b[:i]:
            overlap_length = i

    # 合并字符串
    return b[overlap_length:]


language = set_language()
# language = 'zh'

if language == "zh":
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        #   spk_model="cam++",
        disable_update=True,
    )
else:
    if language != "en":
        language = "auto"
    model = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0", disable_update=True)


def record():
    FORMAT = pyaudio.paInt16  # 音频格式
    CHANNELS = 1  # 通道数（立体声）
    RATE = 16000  # 采样率
    CHUNK = 1024  # 缓冲区大小
    RECORD_SECONDS = 5  # 每次录音的时长
    OVERLAP = 0.5
    WAV_OUTPUT_FILENAME = "output_"  # 输出文件名前缀
    # 初始化PyAudio
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    device = 1
    if language == "zh":
        for i in range(numdevices):
            t = audio.get_device_info_by_host_api_device_index(0, i)
            if (
                t.get("maxInputChannels", 0) > 0
                and t["name"] == "Line 1 (Virtual Audio Cable)"
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
        n = int(RATE * RECORD_SECONDS / CHUNK)
        k = int((OVERLAP / RECORD_SECONDS) * min(len(frames), n))
        frames = frames[-k:]
        for _ in range(n):
            data = stream.read(CHUNK)
            if np.frombuffer(data, np.short).max() < 5:
                continue
            frames.append(data)
        if frames:
            # 保存WAV文件
            timestamp = int(time.time())
            wav_filename = f"{WAV_OUTPUT_FILENAME}{timestamp}.wav"
            wf = wave.open(wav_filename, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()

            # print(f"保存为: {wav_filename}")
            waiting.put(wav_filename)


def generate():
    dt = datetime.now()
    filename = f"transcribe/FunASRTranscribe_{dt.strftime('%Y-%m-%d_%H%M%S')}.txt"
    last_s = ""
    cnt = 0
    while True:
        wav_file = waiting.get()
        res = model.generate(
            input=wav_file,
            batch_size_s=300,
            hotword="雨纪 ameki michiya 维阿 罐罐 百舰",
            disable_pbar=True,
            language=language,
            use_itn=True,
            ban_emo_unk=True,
            batch_size=16,
        )

        if len(res) and res[0]["text"]:
            text = "".join([i["text"] for i in res])
            res = rich_transcription_postprocess(text)
            s = merge_strings(last_s, res[:-1])
            last_s += s
            cnt += len(s)
            print(s, end="")
            last_s = last_s[-20:]
            with open(filename, "a", encoding="utf-8") as f:
                f.write(s)
                if cnt > 15:
                    print()
                    f.write("\n")
                    cnt = 0
        os.remove(wav_file)


t1 = Thread(target=record)
t2 = Thread(target=generate)
t1.start()
t2.start()
t1.join()
t2.join()
