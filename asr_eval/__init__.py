"""ASR 评测工具：数据集加载、模型封装、文本标准化与 CER 统计"""

from .datasets import (
    AudioFilesDataset,
    BaseASRDataset,
    HuggingFaceDataset,
    HuggingFaceDatasetLocal,
)
from .metrics import cer_from_counts, count_edits

from .runner import evaluate_asr
from .text import TextNormalizer

__all__ = [
    "BaseASRDataset",
    "AudioFilesDataset",
    "HuggingFaceDataset",
    "HuggingFaceDatasetLocal",
    "TextNormalizer",
    "count_edits",
    "cer_from_counts",
    "evaluate_asr",
]
