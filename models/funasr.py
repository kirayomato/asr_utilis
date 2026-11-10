import numpy as np
from typing import Any, List

from .base import BaseASRModel


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


class FunASRModel(BaseASRModel):
    """FunASR模型"""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id)
        self.kwargs = kwargs

    def load(self):
        from funasr import AutoModel

        self.model = AutoModel(model=self.model_id, **self.kwargs)

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        res = self.model.generate(input=audio_batch)
        return [item["text"] for item in res]


class paraformer_onnx(BaseASRModel):
    """FunASR ONNX 版 SeacoParaformer"""

    def __init__(self, model_id: str, quantize=True, hotwords: str = ""):
        super().__init__(model_id)
        self.quantize = quantize
        self.hotwords = hotwords
        # "manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
        # "manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline"

    def load(self):
        from funasr_onnx import SeacoParaformer
        import funasr_onnx

        funasr_onnx.utils.postprocess_utils.abbr_dispose = abbr_dispose
        self.model = SeacoParaformer(
            self.model_id, batch_size=1, quantize=self.quantize
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        results = [
            "".join(item["preds"] for item in self.model(audio, self.hotwords))
            for audio in audio_batch
        ]
        return results
