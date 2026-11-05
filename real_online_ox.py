from typing import Any, List
from funasr_onnx.paraformer_online_bin import Paraformer
import funasr_onnx
from datetime import datetime
from queue import Queue
from threading import Thread
import numpy as np
import pyaudio


def abbr_dispose(words: List[Any], time_stamp: List[List] = None) -> List[Any]:
    words_size = len(words)
    word_lists = []
    abbr_begin = []
    abbr_end = []
    last_num = -1
    ts_lists = []
    ts_nums = []
    ts_index = 0
    for num in range(words_size):
        if num <= last_num:
            continue

        if len(words[num]) == 1 and words[num].encode("utf-8").isalpha():
            if (
                num + 1 < words_size
                and words[num + 1] == " "
                and num + 2 < words_size
                and len(words[num + 2]) == 1
                and words[num + 2].encode("utf-8").isalpha()
            ):
                # found the begin of abbr
                abbr_begin.append(num)
                num += 2
                abbr_end.append(num)
                # to find the end of abbr
                while True:
                    num += 1
                    if num < words_size and words[num] == " ":
                        num += 1
                        if (
                            num < words_size
                            and len(words[num]) == 1
                            and words[num].encode("utf-8").isalpha()
                        ):
                            abbr_end.pop()
                            abbr_end.append(num)
                            last_num = num
                        else:
                            break
                    else:
                        break

    for num in range(words_size):
        if words[num] == " ":
            ts_nums.append(ts_index)
        else:
            ts_nums.append(ts_index)
            ts_index += 1
    last_num = -1
    for num in range(words_size):
        if num <= last_num:
            continue

        if num in abbr_begin:
            if time_stamp is not None:
                begin = time_stamp[ts_nums[num]][0]
            word = words[num].upper()
            num += 1
            while num < words_size:
                if num in abbr_end:
                    word += words[num].upper()
                    last_num = num
                    break
                else:
                    if words[num].encode("utf-8").isalpha():
                        word += words[num].upper()
                num += 1
            word_lists.append(word)
            if time_stamp is not None:
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
        else:
            word_lists.append(words[num])
            if time_stamp is not None and words[num] != " ":
                begin = time_stamp[ts_nums[num]][0]
                end = time_stamp[ts_nums[num]][1]
                ts_lists.append([begin, end])
                begin = end

    if time_stamp is not None:
        return word_lists, ts_lists
    else:
        return word_lists


funasr_onnx.utils.postprocess_utils.abbr_dispose = abbr_dispose


def set_device():
    # try:
    #     return func_timeout(5, lambda: input("device:").lower())
    # except:
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


chunck = [8, 16, 8]


def record():
    FORMAT = pyaudio.paInt16  # 音频格式
    CHANNELS = 1  # 通道数（立体声）
    RATE = 16000  # 采样率
    CHUNK = 1000  # 缓冲区大小
    RECORD_SECONDS = chunck[1] * 0.06  # 每次录音的时长
    OVERLAP = 0.25
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
    while True:
        frames = []
        # 录制音频
        n = int(RATE * RECORD_SECONDS / CHUNK)
        # k = int((OVERLAP / RECORD_SECONDS) * min(len(frames), n))
        # frames = frames[-k:]
        cnt = 0
        for _ in range(n):
            data = stream.read(CHUNK)
            frames.append(data)

        if len(frames):
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
            waiting.put(waveform)


def generate():
    dt = datetime.now()
    filename = f"transcribe/transcribe_{dt.strftime('%Y-%m-%d_%H%M%S')}.txt"
    last_s = ""
    cnt = 0
    while True:
        wav_file = waiting.get()
        try:
            # t0 = time()
            res = model(wav_file)
            if isinstance(res, list):
                try:
                    res = res[0]["preds"][0]
                except:
                    continue
            # print(f"{time()-t0:.2f}s")
            if len(res):
                s = merge_strings(last_s, res[:-1])
                last_s += s
                cnt += len(s)
                print(s, end="", flush=True)
                last_s = last_s[-20:]
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(s)
                    if cnt > 15:
                        print()
                        f.write("\n")
                        cnt = 0
        except BaseException as e:
            print(e)
        # os.remove(wav_file)


model_dir = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
model = Paraformer(model_dir, chunk_size=chunck, batch_size=1, quantize=True)

t1 = Thread(target=record)
t2 = Thread(target=generate)
t1.start()
t2.start()
t1.join()
t2.join()
