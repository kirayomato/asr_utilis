import librosa
import numpy as np
from pathlib import Path
from typing import Any, Iterator, List, Tuple
from tqdm import tqdm


class BaseASRDataset:
    """ASR数据集基类"""

    def __init__(self, name: str):
        self.name = name
        self.audio_data = []
        self.transcripts = []
        self._loaded = False

    def load(self, **kwargs) -> Tuple[List[np.ndarray], List[str]]:
        """加载数据集"""
        raise NotImplementedError

    def get_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """获取已加载的数据，如果未加载则自动加载"""
        if not self._loaded:
            self.load()
            self._loaded = True
        return self.audio_data, self.transcripts

    def is_loaded(self) -> bool:
        """检查数据是否已加载"""
        return self._loaded

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return self.audio_data[idx], self.transcripts[idx]


class AudioFilesDataset(BaseASRDataset):
    """音频文件数据集"""

    def __init__(
        self, data_dir: str, extensions: List[str] = None, max_samples: int = -1
    ):
        super().__init__("audio_files")
        self.data_dir = Path(data_dir)
        self.extensions = extensions or [".wav", ".mp3", ".flac"]
        self.max_samples = max_samples

    def load(self) -> Tuple[List[np.ndarray], List[str]]:
        """从目录加载音频文件"""
        audio_files = []
        for ext in self.extensions:
            audio_files.extend(list(self.data_dir.glob(f"*{ext}")))

        if self.max_samples > 0:
            audio_files = audio_files[: self.max_samples]

        self.audio_data = []
        self.transcripts = []

        # 尝试加载转录文件
        transcript_file = self.data_dir / "transcript.txt"
        transcript_map = {}

        if transcript_file.exists():
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        transcript_map[parts[0]] = parts[1]

        for audio_file in tqdm(audio_files, desc=f"Loading {self.name}"):
            try:
                audio, _ = librosa.load(audio_file, sr=16000)
                self.audio_data.append(audio)

                # 获取转录文本
                transcript = transcript_map.get(audio_file.name, "")
                self.transcripts.append(transcript)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue

        return self.audio_data, self.transcripts


def _iter_samples(dataset: Any, desc: str) -> Iterator[Tuple[np.ndarray, str]]:
    """逐条解析样本中的音频数组与转录文本，字段名按常见的几种命名猜测"""
    for item in tqdm(dataset, desc=desc):
        try:
            # 获取音频数据
            if "array" in item:
                audio = item["array"]
            elif "audio" in item:
                audio = item["audio"]["array"]
            else:
                audio_fields = [k for k in item.keys() if "audio" in k.lower()]
                if not audio_fields:
                    continue
                audio = item[audio_fields[0]]["array"]

            # 获取转录文本
            text_fields = [
                k
                for k in item.keys()
                if "text" in k.lower()
                or "transcript" in k.lower()
                or "sentence" in k.lower()
            ]
            transcript = item[text_fields[0]] if text_fields else ""

            yield audio, transcript
        except Exception as e:
            print(f"Error loading sample: {e}")
            continue


class HuggingFaceDataset(BaseASRDataset):
    """HuggingFace数据集"""

    def __init__(self, dataset_name: str, split: str = "test", max_samples: int = -1):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples

    def load(self) -> Tuple[List[np.ndarray], List[str]]:
        """从HuggingFace加载数据集"""
        from datasets import load_dataset

        dataset = load_dataset(self.dataset_name, split=self.split)

        if self.max_samples > 0:
            dataset = dataset.select(range(self.max_samples))

        self.audio_data = []
        self.transcripts = []

        for audio, transcript in _iter_samples(dataset, f"Loading {self.name}"):
            self.audio_data.append(audio)
            self.transcripts.append(transcript)

        return self.audio_data, self.transcripts


class HuggingFaceDatasetLocal(BaseASRDataset):
    """HuggingFace本地数据集"""

    def __init__(self, dataset_name: str, split: str = "test", max_samples: int = -1):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples

    def load(self) -> Tuple[List[np.ndarray], List[str]]:
        """从本地磁盘加载HuggingFace数据集"""
        from datasets import load_from_disk

        dataset = load_from_disk(self.dataset_name + "/" + self.split)

        if self.max_samples > 0:
            dataset = dataset.select(range(self.max_samples))

        self.audio_data = []
        self.transcripts = []

        for audio, transcript in _iter_samples(dataset, f"Loading {self.name}"):
            self.audio_data.append(audio)
            self.transcripts.append(transcript)

        return self.audio_data, self.transcripts
