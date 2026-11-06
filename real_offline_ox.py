from typing import Any, List
from funasr_onnx import SeacoParaformer
import funasr_onnx
from realtime_speech import start_transcribe
import numpy as np


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

model_dir = "manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
model = SeacoParaformer(model_dir, batch_size=1, quantize=True)


def transcribe(wav: np.ndarray) -> str:
    res = model(
        wav,
        hotwords="雨纪 ameki 米汀 米米 维阿 罐罐 百舰 舰长 点酥 酥酥 海海 栞栞 岁己",
    )
    res = "".join([i["preds"] for i in res])
    return res


start_transcribe(transcribe, timing=False)
