import numpy as np
import torch
from abc import ABC, abstractmethod
from math import ceil
from tqdm import tqdm
from typing import List, Optional


class BaseASRModel(ABC):
    """ASR模型基类"""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None

    @abstractmethod
    def load(self):
        """加载模型"""
        pass

    @abstractmethod
    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        """转录一批音频"""
        pass

    def batch_transcribe(
        self,
        audio_data: List[np.ndarray],
        batch_size: int = 32,
        progress_bar: bool = True,
        loaded_data: Optional[List[np.ndarray]] = None,
    ) -> List[str]:
        """分批转录所有音频"""
        self.load()

        # 如果已提供处理好的音频数据，则直接使用
        if loaded_data is not None:
            audio_data = loaded_data

        predictions = []
        n_batches = ceil(len(audio_data) / batch_size)

        iterator = range(n_batches)
        if progress_bar:
            iterator = tqdm(iterator, desc="Transcribing")

        for i in iterator:
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            audio_batch = audio_data[batch_start:batch_end]

            # 填充到相同长度
            max_length = max([len(data) for data in audio_batch])
            padded_batch = [
                np.pad(data, (0, max_length - len(data)), "constant")
                for data in audio_batch
            ]

            batch_predictions = self.transcribe(padded_batch)
            predictions.extend(batch_predictions)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions
