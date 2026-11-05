import funasr_onnx
from funasr_onnx.utils.timestamp_utils import time_stamp_lfr6_onnx
from funasr_onnx.utils.postprocess_utils import (
    sentence_postprocess_sentencepiece,
    isAllAlpha,
    isAllChinese,
)
from funasr_onnx.utils.utils import ONNXRuntimeError
from funasr_onnx import SeacoParaformer, Fsmn_vad
import numpy as np
import copy
from typing import Any, List, Union
from datetime import datetime, timezone
from os.path import splitext, basename
from os import system, rename, remove
from time import time
from threading import Thread
from colorama import Fore, init, Style
from tqdm import tqdm
import traceback

# if splitext(video_path)[1] != 'wav':
#     path1 = video_path.replace(' ', '_')
#     rename(video_path, path1)
#     path1 = splitext(path1)[0]+'.wav'
#     system(f'ffmpeg -i {video_path} {path1}')
#     video_path = path1
video = []


def deal_input():
    while 1:
        video.append(input())


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


def sentence_postprocess(words: List[Any], time_stamp: List[List] = None):
    middle_lists = []
    word_lists = []
    word_item = ""
    ts_lists = []

    # wash words lists
    for i in words:
        word = ""
        if isinstance(i, str):
            word = i
        else:
            word = i.decode("utf-8")

        if word in ["<s>", "</s>", "<unk>"]:
            continue
        else:
            middle_lists.append(word)

    # all chinese characters
    if isAllChinese(middle_lists):
        for i, ch in enumerate(middle_lists):
            word_lists.append(ch.replace(" ", ""))
        if time_stamp is not None:
            ts_lists = time_stamp

    # all alpha characters
    elif isAllAlpha(middle_lists):
        ts_flag = True
        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ""
            if "@@" in ch:
                word = ch.replace("@@", "")
                word_item += word
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            else:
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1]
                    ts_lists.append([begin, end])
                    begin = end

    # mix characters
    else:
        alpha_blank = False
        ts_flag = True
        begin = -1
        end = -1
        for i, ch in enumerate(middle_lists):
            if ts_flag and time_stamp is not None:
                begin = time_stamp[i][0]
                end = time_stamp[i][1]
            word = ""
            if isAllChinese(ch):
                if alpha_blank is True:
                    word_lists.pop()
                word_lists.append(ch)
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = True
                    ts_lists.append([begin, end])
                    begin = end
            elif "@" in ch:
                word = ch.replace("@", "")
                word_item += word
                alpha_blank = False
                if time_stamp is not None:
                    ts_flag = False
                    end = time_stamp[i][1]
            elif isAllAlpha(ch):
                word_item += ch
                word_lists.append(word_item)
                word_lists.append(" ")
                word_item = ""
                alpha_blank = True
                if time_stamp is not None:
                    ts_flag = True
                    end = time_stamp[i][1]
                    ts_lists.append([begin, end])
                    begin = end
            else:
                raise ValueError("invalid character: {}".format(ch))

    if time_stamp is not None:
        word_lists, ts_lists = abbr_dispose(word_lists, ts_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != " ":
                real_word_lists.append(ch)
        sentence = " ".join(real_word_lists).strip()
        return sentence, ts_lists, real_word_lists
    else:
        word_lists = abbr_dispose(word_lists)
        real_word_lists = []
        for ch in word_lists:
            if ch != " ":
                real_word_lists.append(ch)
        sentence = "".join(word_lists).strip()
        return sentence, real_word_lists


funasr_onnx.utils.postprocess_utils.abbr_dispose = abbr_dispose


def generate_with_vad(
    self: SeacoParaformer,
    wav_content: np.ndarray,
    hotwords: str,
    vad: Fsmn_vad = None,
    batch_size=1,
    **kwargs,
) -> List:
    # make hotword list
    hotwords, hotwords_length = self.proc_hotword(hotwords)
    [bias_embed] = self.eb_infer(hotwords, hotwords_length)
    # index from bias_embed
    bias_embed = bias_embed.transpose(1, 0, 2)
    _ind = np.arange(0, len(hotwords)).tolist()
    bias_embed = bias_embed[_ind, hotwords_length.tolist()]
    bias_embed = np.expand_dims(bias_embed, axis=0)
    waveform_list = []
    if vad:
        # scale = int(len(waveform_list[0]) / vad[-1][-1])
        scale = 16
        for [i, j] in vad:
            waveform_list.append(wav_content[i * scale : j * scale])
    waveform_nums = len(waveform_list)
    asr_res = []

    for beg_idx in tqdm(range(0, waveform_nums, batch_size)):
        end_idx = min(waveform_nums, beg_idx + batch_size)
        feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
        try:
            outputs = self.bb_infer(feats, feats_len, bias_embed)
            am_scores, valid_token_lens = outputs[0], outputs[1]

            if len(outputs) == 4:
                # for BiCifParaformer Inference
                _, us_peaks = outputs[2], outputs[3]
            else:
                _, us_peaks = None, None

        except ONNXRuntimeError:
            print("input wav is silence or noise")
            preds = [""]
        else:
            preds = self.decode(am_scores, valid_token_lens)
            if us_peaks is None:
                for pred in preds:
                    if self.language == "en-bpe":
                        pred = sentence_postprocess_sentencepiece(pred)
                    else:
                        pred = sentence_postprocess(pred)
                    asr_res.append({"preds": pred})
            else:
                try:
                    for pred, us_peaks_ in zip(preds, us_peaks):
                        raw_tokens = pred
                        timestamp, timestamp_raw = time_stamp_lfr6_onnx(
                            us_peaks_, copy.copy(raw_tokens)
                        )
                        text_proc, timestamp_proc, _ = sentence_postprocess(
                            raw_tokens, timestamp_raw
                        )
                        # logging.warning(timestamp)
                        if len(self.plot_timestamp_to):
                            self.plot_wave_timestamp(
                                waveform_list[0], timestamp, self.plot_timestamp_to
                            )
                        asr_res.append(
                            {
                                "preds": text_proc,
                                "timestamp": timestamp_proc,
                                "raw_tokens": raw_tokens,
                            }
                        )
                except:
                    continue
    if vad:
        res = {
            "preds": "",
            "timestamp": [],
            "raw_tokens": [],
        }
        for idx in range(len(asr_res)):
            m = len(asr_res[idx]["timestamp"])
            for i in range(m):
                asr_res[idx]["timestamp"][i][0] += vad[idx][0]
                asr_res[idx]["timestamp"][i][1] += vad[idx][0]
            res["preds"] += asr_res[idx]["preds"] + " "
            res["timestamp"] += asr_res[idx]["timestamp"]
            res["raw_tokens"] += asr_res[idx]["raw_tokens"]
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
    audio = model.load_data(video_path, 16000)[0]

    ouput_path = splitext(video_path)[0] + ".srt"
    raw = splitext(video_path)[0] + "_fun.txt"

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
            dt = datetime.fromtimestamp(ts[i][0] / 1000, timezone.utc)
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
            dt = datetime.fromtimestamp(ts[i - 1][1] / 1000, timezone.utc)
            f.write(dt.strftime("%H:%M:%S,%f") + "\n")
            f.write(s + "\n\n")
            s = ""
            j += 1
            cnt += 1

    print(f"Convert:{time()-t0:.2f}s")
    system(f'"{ouput_path}"')


if __name__ == "__main__":
    init()
    t1 = Thread(target=deal_input)
    t1.start()
    t0 = time()
    from funasr_onnx import SeacoParaformer, CT_Transformer, Fsmn_vad

    model_dir = "manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline"
    model = SeacoParaformer(model_dir, batch_size=1, quantize=True)
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
