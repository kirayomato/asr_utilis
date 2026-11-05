from datetime import datetime
import os
from queue import Queue
from threading import Thread
import numpy as np
from func_timeout import func_timeout
import pyaudio
import wave
from time import time


def set_device():
    try:
        return func_timeout(5, lambda: input("device:").lower())
    except:
        return "1"


waiting = Queue()
dev = set_device()


def merge_strings(a, b):
    max_length = min(len(a), len(b))
    overlap_length = 0

    # 找到最长的重叠部分
    for i in range(1, max_length + 1):
        if a[-i:] == b[:i]:
            overlap_length = i

    # 合并字符串
    return b[overlap_length:]


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
        n = int(RATE * RECORD_SECONDS / CHUNK)
        k = int((OVERLAP / RECORD_SECONDS) * min(len(frames), n))
        frames = frames[-k:]
        for _ in range(n):
            data = stream.read(CHUNK)
            if np.frombuffer(data, np.short).max() < 5:
                continue
            frames.append(data)
        if len(frames):
            # 保存WAV文件
            # timestamp = int(time())
            # wav_filename = f"{WAV_OUTPUT_FILENAME}{timestamp}.wav"
            # wf = wave.open(wav_filename, "wb")
            # wf.setnchannels(CHANNELS)
            # wf.setsampwidth(audio.get_sample_size(FORMAT))
            # wf.setframerate(RATE)
            # wf.writeframes(b"".join(frames))
            # wf.close()

            # # print(f"保存为: {wav_filename}")
            # waiting.put(wav_filename)
            waveform = (
                np.frombuffer(b"".join(frames), dtype=np.short).astype(np.float32)
                / 32768.0
            )
            waiting.put(waveform)


def generate():
    dt = datetime.now()
    filename = f"transcribe/transcribe_{dt.strftime('%Y-%m-%d_%H%M%S')}.txt"
    last_s = ""
    cnt = 0
    while True:
        wav_file = waiting.get()
        try:
            t0 = time()
            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|zh|>"
            inputs = processor(
                text=prompt, audios=wav_file, return_tensors="pt", sampling_rate=16_000
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            generate_ids = model.generate(**inputs, max_length=512)
            generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]

            res = (
                processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                .replace(" ", "")
                .replace("\n", "")
            )

            print(f"{time()-t0:.2f}s")

            if len(res):
                s = merge_strings(last_s, res[:-1])
                last_s += s
                cnt += len(s)
                # s = res
                # cnt = 100
                print(s, end="")
                last_s = last_s[-20:]
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(s)
                    if cnt > 15:
                        print()
                        f.write("\n")
                        cnt = 0
        except BaseException as e:
            print(e)
            # print(input_tokens, output_tokens, audio_tokens)
        # os.remove(wav_file)


from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
)

processor = AutoProcessor.from_pretrained("alicekyting/Qwen2-Audio-7B-Instruct-4bit")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "alicekyting/Qwen2-Audio-7B-Instruct-4bit", device_map="auto"
)


t1 = Thread(target=record)
t2 = Thread(target=generate)
t1.start()
t2.start()
t1.join()
t2.join()
