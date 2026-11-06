import torch
import librosa
import numpy as np
import jiwer
import re
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, Optional
from time import time
from math import ceil
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import soundfile as sf


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

        for item in tqdm(dataset, desc=f"Loading {self.name}"):
            try:
                # 获取音频数据
                if "array" in item:
                    audio = item["array"]
                elif "audio" in item:
                    audio = item["audio"]["array"]
                else:
                    # 尝试其他常见字段
                    audio_fields = [k for k in item.keys() if "audio" in k.lower()]
                    if audio_fields:
                        audio = item[audio_fields[0]]["array"]
                    else:
                        continue

                # 获取转录文本
                text_fields = [
                    k
                    for k in item.keys()
                    if "text" in k.lower()
                    or "transcript" in k.lower()
                    or "sentence" in k.lower()
                ]
                if text_fields:
                    transcript = item[text_fields[0]]
                else:
                    transcript = ""

                self.audio_data.append(audio)
                self.transcripts.append(transcript)
            except Exception as e:
                print(f"Error loading sample: {e}")
                continue

        return self.audio_data, self.transcripts


class HuggingFaceDatasetLocal(BaseASRDataset):
    """HuggingFace数据集"""

    def __init__(self, dataset_name: str, split: str = "test", max_samples: int = -1):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples

    def load(self) -> Tuple[List[np.ndarray], List[str]]:
        """从HuggingFace加载数据集"""
        from datasets import load_from_disk

        dataset = load_from_disk(self.dataset_name + "/" + self.split)

        if self.max_samples > 0:
            dataset = dataset.select(range(self.max_samples))

        self.audio_data = []
        self.transcripts = []

        for item in tqdm(dataset, desc=f"Loading {self.name}"):
            try:
                # 获取音频数据
                if "array" in item:
                    audio = item["array"]
                elif "audio" in item:
                    audio = item["audio"]["array"]
                else:
                    # 尝试其他常见字段
                    audio_fields = [k for k in item.keys() if "audio" in k.lower()]
                    if audio_fields:
                        audio = item[audio_fields[0]]["array"]
                    else:
                        continue

                # 获取转录文本
                text_fields = [
                    k
                    for k in item.keys()
                    if "text" in k.lower()
                    or "transcript" in k.lower()
                    or "sentence" in k.lower()
                ]
                if text_fields:
                    transcript = item[text_fields[0]]
                else:
                    transcript = ""

                self.audio_data.append(audio)
                self.transcripts.append(transcript)
            except Exception as e:
                print(f"Error loading sample: {e}")
                continue

        return self.audio_data, self.transcripts


class TextNormalizer:
    """文本标准化器"""

    def __init__(
        self,
        remove_punctuation: bool = True,
        lower_case: bool = True,
        separate_chinese_english: bool = True,
        custom_fn: Optional[Callable[[str], str]] = None,
    ):
        self.remove_punctuation = remove_punctuation
        self.lower_case = lower_case
        self.separate_chinese_english = separate_chinese_english
        self.custom_fn = custom_fn

    def normalize(self, text: str) -> str:
        """标准化文本"""
        if self.custom_fn:
            return self.custom_fn(text)

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        if self.separate_chinese_english:
            parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text)
            result = []
            for word in parts:
                if word.encode("UTF-8").isalpha():
                    result.append(word.lower() if self.lower_case else word)
                else:
                    result.extend(list(word))
            text = " ".join(result)
        elif self.lower_case:
            text = text.lower()

        return text.strip()


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


class WhisperModel(BaseASRModel):
    """Whisper模型"""

    def __init__(self, model_id: str, device: str = None):
        super().__init__(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        import whisper

        self.model = whisper.load_model(self.model_id, device=self.device)

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        results = []
        for audio in audio_batch:
            audio_tensor = torch.from_numpy(audio).float()
            result = self.model.transcribe(audio_tensor)
            results.append(result["text"])
        return results


class Wav2Vec2Model(BaseASRModel):
    """Wav2Vec2模型"""

    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        # 预处理音频
        inputs = self.processor(
            audio_batch, sampling_rate=16000, return_tensors="pt", padding=True
        )

        # 移动到GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # 解码
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)

        return transcriptions


class paraformer_onnx(BaseASRModel):

    def __init__(self, model_id: str, quantize=True):
        super().__init__(model_id)
        self.quantize = quantize
        # "manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
        # "manyeyes/paraformer-seaco-large-zh-timestamp-onnx-offline"

    def load(self):
        from funasr_onnx import SeacoParaformer

        self.model = SeacoParaformer(
            self.model_id, batch_size=1, quantize=self.quantize
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        results = [self.model(audio, "")[0]["preds"] for audio in audio_batch]
        return results


class zipformer(BaseASRModel):

    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self):
        import sherpa_onnx

        if self.model_id == "online_ctc":
            path = "./sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "offline_ctc":
            path = "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"

            self.model = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "online_transducer":
            path = "./sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30/"
            tokens = path + "tokens.txt"
            encoder = path + "encoder.int8.onnx"
            decoder = path + "decoder.onnx"
            joiner = path + "joiner.int8.onnx"
            self.model = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                decoding_method="modified_beam_search",
                num_threads=8,
                # hotwords_file=args.hotwords_file,
            )
        elif self.model_id == "dolphin":
            path = "./sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "omnilingual":
            path = (
                "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/"
            )
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "offline_wenet_wenetspeech":
            path = "./sherpa-onnx-zh-wenet-wenetspeech/"
            model = path + "model.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )
        elif self.model_id == "online_wenet_wenetspeech":
            path = "./sherpa-onnx-zh-wenet-wenetspeech/"
            model = path + "model-streaming.int8.onnx"
            tokens = path + "tokens.txt"
            self.model = sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
                model=model,
                tokens=tokens,
                num_threads=8,
            )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        streams = []
        for wav in audio_batch:
            stream = self.model.create_stream()
            stream.accept_waveform(16000, wav)
            if isinstance(self.model, sherpa_onnx.OnlineRecognizer):
                tail_paddings = np.zeros(int(0.66 * 16000), dtype=np.float32)
                stream.accept_waveform(16000, tail_paddings)
                stream.input_finished()
            streams.append(stream)
        if isinstance(self.model, sherpa_onnx.OnlineRecognizer):
            while True:
                ready_list = []
                for s in streams:
                    if self.model.is_ready(s):
                        ready_list.append(s)
                if len(ready_list) == 0:
                    break
                self.model.decode_streams(ready_list)
            return [self.model.get_result(s) for s in streams]
        else:
            for stream in streams:
                self.model.decode_stream(stream)
            return [s.result.text for s in streams]


class Qwen3ASRModel(BaseASRModel):
    """Qwen3ASR模型"""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id)
        self.kwargs = kwargs

    def load(self):
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            f"Qwen/{self.model_id}",
            # r"F:\Program\cache\modelscope\models\Qwen\Qwen3-ASR-0___6B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32,  # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256,  # Maximum number of tokens to generate. Set a larger value for long audio input.
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        audio = [(i, 16000) for i in audio_batch]
        res = self.model.transcribe(audio=audio, language="Chinese")
        return [item.text for item in res]


class Nemotron(BaseASRModel):

    def __init__(self, model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"):
        super().__init__(model_id)

    def load(self):
        from transformers import AutoModelForRNNT, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForRNNT.from_pretrained(self.model_id, device_map="auto")

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        result = []
        for wav in audio_batch:
            inputs = self.processor(
                wav,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                language="zh-CN",
            )
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            text, ts = self.processor.decode(
                output.sequences, durations=output.durations, skip_special_tokens=True
            )
            result.append(text[0])
        return result


class ark_audio_onnx(BaseASRModel):

    def __init__(self, model_id: str):
        super().__init__(model_id)

    def load(self):
        os.chdir("ark-asr-0.6b-int8-onnx")
        from infer_ark_audio_onnx import ArkAsrOnnxRuntime

        self.model = ArkAsrOnnxRuntime(Path("."))

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        result = []
        for wav in audio_batch:
            sf.write("soundfile.wav", wav, 16000, subtype="PCM_16")
            text = self.model.transcribe(
                audio_path="soundfile.wav",
                max_new_tokens=128,
                max_audio_seconds=30,
                precision="int8",
                asr_block_token_id_from=151670,
            )
            result.append(text)
        return result


class GLMASRLLMModel(BaseASRModel):
    """GLMASRLLMModel模型"""

    def __init__(self, model_id: str = "GLM-ASR-Nano-2512", **kwargs):
        super().__init__(model_id)
        self.kwargs = kwargs

    def load(self):
        from transformers import AutoModel, AutoProcessor

        repo_id = f"zai-org/{self.model_id}"
        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = AutoModel.from_pretrained(
            repo_id, dtype=torch.bfloat16, device_map="auto"
        )

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        inputs = self.processor.apply_transcription_request(audio_batch)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=500, pad_token_id=59246
        )
        decoded_outputs = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded_outputs


def evaluate_asr(
    dataset: BaseASRDataset,
    model: BaseASRModel,
    batch_size: int = 32,
    normalizer: Optional[TextNormalizer] = None,
    save_results: bool = True,
    output_dir: str = "results",
    progress_bar: bool = True,
    suffix: str = "",
    loaded_data: Optional[Tuple[List[np.ndarray], List[str]]] = None,
) -> Dict[str, Any]:
    """
    评估ASR模型

    Args:
        dataset: ASR数据集
        model: ASR模型
        batch_size: 批处理大小
        normalizer: 文本标准化器
        save_results: 是否保存结果
        output_dir: 输出目录
        progress_bar: 是否显示进度条
        loaded_data: 已加载的数据 (audio_data, targets)，如果为None则自动加载

    Returns:
        评估结果字典
    """
    # 如果已提供加载好的数据，则直接使用，否则加载数据集
    if loaded_data is not None:
        audio_data, targets = loaded_data
        print(f"使用已加载的数据集: {dataset.name} (样本数: {len(audio_data)})")
    else:
        print(f"加载数据集: {dataset.name}")
        audio_data, targets = dataset.get_data()

        if not audio_data or not targets:
            raise ValueError("数据集为空或加载失败")

        print(f"数据集大小: {len(audio_data)} 个样本")

    print(f"加载模型: {model.model_id}")
    t0 = time()
    # 转录音频
    predictions = model.batch_transcribe(
        audio_data, batch_size=batch_size, progress_bar=progress_bar
    )
    t0 = time() - t0
    # 确保预测数量匹配
    if len(predictions) != len(targets):
        print(f"警告: 预测数量({len(predictions)})与目标数量({len(targets)})不匹配")
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]

    # 文本标准化
    normalizer = normalizer or TextNormalizer()
    normalized_targets = [normalizer.normalize(t) for t in targets]
    normalized_preds = [normalizer.normalize(p) for p in predictions]

    # 计算CER
    print("计算评估指标...")

    # 逐样本计算
    H, S, D, I = 0, 0, 0, 0
    for truth, hypo in zip(normalized_targets, normalized_preds):
        try:
            chunk_metrics = jiwer.process_words(truth, hypo)
            H += chunk_metrics.hits
            S += chunk_metrics.substitutions
            D += chunk_metrics.deletions
            I += chunk_metrics.insertions
        except Exception as e:
            print(f"处理错误: {e}\n源文本: {truth}\n目标文本: {hypo}")

    cer = (S + D + I) / (H + S + D) if (H + S + D) > 0 else 0

    # 准备结果
    results = {
        "model": model.model_id,
        "dataset": dataset.name,
        "samples": len(audio_data),
        "cer": cer,
        "cer_percentage": cer * 100,
        "hits": H,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
    }

    # 保存结果
    if save_results:
        output_dir = Path(output_dir + "/" + dataset.name)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 保存评估结果
        result_file = (
            output_dir / f"{model.model_id.replace('/', '_')}_{suffix}results.txt"
        )
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"模型: {model.model_id}\n")
            f.write(f"数据集: {dataset.name}\n")
            f.write(f"样本数: {len(audio_data)}\n")
            f.write(f"batch size:{batch_size}\n")
            f.write(f"耗时: {t0:.2f}s，速度：{len(audio_data)/t0:.2f}it/s\n")
            f.write("-" * 50 + "\n")
            f.write(f"CER: {cer:.4f} ({cer*100:.2f}%)\n")
            f.write(f"命中(H): {H}\n")
            f.write(f"替换(S): {S}\n")
            f.write(f"删除(D): {D}\n")
            f.write(f"插入(I): {I}\n")
            f.write("-" * 50 + "\n")

            # 保存前10个样本的详细结果
            f.write("\n前10个样本的详细结果:\n")
            for i in range(min(10, len(targets))):
                f.write(f"\n样本 {i+1}:\n")
                f.write(f"  目标: {targets[i]}\n")
                f.write(f"  预测: {predictions[i]}\n")
                f.write(f"  标准化目标: {normalized_targets[i]}\n")
                f.write(f"  标准化预测: {normalized_preds[i]}\n")

        # 保存所有预测结果
        pred_file = output_dir / f"{model.model_id.replace('/', '_')}_predictions.tsv"
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(
                "index\ttarget\tprediction\ttarget_normalized\tprediction_normalized\n"
            )
            for i, (target, pred, norm_t, norm_p) in enumerate(
                zip(targets, predictions, normalized_targets, normalized_preds)
            ):
                f.write(f"{i}\t{target}\t{pred}\t{norm_t}\t{norm_p}\n")

        print(f"结果已保存到: {result_file}")
        print(f"预测结果已保存到: {pred_file}")

    return results


# 使用示例
if __name__ == "__main__":

    # 创建数据集
    # dataset = CommonVoiceDataset(
    #     data_dir="D:/Program/VLM/whisper/zh-CN_test_0/",
    #     max_samples=-1,  # 只测试100个样本
    # )
    # dataset = HuggingFaceDatasetLocal("ark_new", "test", max_samples=-1)
    # import os

    # print(os.getenv("HF_HOME"))
    dataset = HuggingFaceDataset("urarik/arknights_voices_zh-processed", split="test")
    # 创建模型列表
    models = [
        # zipformer("online_transducer"),
        # zipformer("online_ctc"),
        # zipformer("offline_ctc"),
        # zipformer("dolphin"),
        # paraformer_onnx("paraformer-zh-onnx"),
        # zipformer("omnilingual"),
        # zipformer("online_wenet_multi_cn"),
        # zipformer("offline_wenet_multi_cn"),
        # zipformer("online_wenet_wenetspeech"),
        # zipformer("offline_wenet_wenetspeech"),
        # paraformer_onnx("paraformer_onnx"),
        Qwen3ASRModel("Qwen3-ASR-0.6B")
    ]

    # 创建文本标准化器
    normalizer = TextNormalizer()

    for model in models:
        try:
            results = evaluate_asr(
                dataset=dataset,
                model=model,
                batch_size=16,
                normalizer=normalizer,
                output_dir="results",
                suffix="new_",
                # save_results=False,
            )
            print(f"CER: {results['cer_percentage']:.2f}%")
        except Exception as e:
            print(f"评估模型 {model.model_id} 失败: {e}")
            continue

    # 方式2: 手动预加载数据并传递（更灵活）
    # print("\n=== 方式2: 手动预加载数据 ===")
    # # 先手动加载数据集
    # audio_data, targets = dataset.load_once()
    # print(f"数据集已预加载，样本数: {len(audio_data)}")

    # # 然后评估多个模型，每次都传递已加载的数据
    # for model in models:
    #     try:
    #         results = evaluate_asr(
    #             dataset=dataset,
    #             model=model,
    #             batch_size=1,
    #             normalizer=normalizer,
    #             output_dir="results",
    #             loaded_data=(audio_data, targets),  # 传递已加载的数据
    #         )
    #         print(f"CER: {results['cer_percentage']:.2f}%")
    #     except Exception as e:
    #         print(f"评估模型 {model.model_id} 失败: {e}")
    #         continue

    # 方式3: 检查数据是否已加载
    # print(f"\n数据集加载状态: {dataset.is_loaded()}")

    # # 示例2: HuggingFace数据集 + Whisper
    # print("\n示例2: HuggingFace数据集 + Whisper")

    # # 创建HuggingFace数据集
    # dataset2 = HuggingFaceDataset(
    #     dataset_name="mozilla-foundation/common_voice_11_0",
    #     split="test",
    #     max_samples=50,
    # )

    # # 创建Whisper模型
    # model2 = WhisperModel("base")

    # # 自定义文本标准化器
    # def custom_normalizer(text: str) -> str:
    #     text = text.lower()
    #     text = re.sub(r"[^\w\s]", "", text)
    #     return text.strip()

    # normalizer2 = TextNormalizer(custom_fn=custom_normalizer)

    # # 评估
    # try:
    #     results2 = evaluate_asr(
    #         dataset=dataset2,
    #         model=model2,
    #         batch_size=16,
    #         normalizer=normalizer2,
    #         output_dir="results",
    #     )
    #     print(f"CER: {results2['cer_percentage']:.2f}%")
    # except Exception as e:
    #     print(f"评估失败: {e}")

    # # 示例3: 音频文件数据集 + Wav2Vec2
    # print("\n示例3: 音频文件数据集 + Wav2Vec2")

    # # 创建音频文件数据集
    # dataset3 = AudioFilesDataset(
    #     data_dir="path/to/your/audio/files/",
    #     extensions=[".wav", ".mp3"],
    #     max_samples=50,
    # )

    # # 创建Wav2Vec2模型
    # model3 = Wav2Vec2Model("facebook/wav2vec2-base-960h")

    # # 评估
    # try:
    #     results3 = evaluate_asr(
    #         dataset=dataset3, model=model3, batch_size=8, output_dir="results"
    #     )
    #     print(f"CER: {results3['cer_percentage']:.2f}%")
    # except Exception as e:
    #     print(f"评估失败: {e}")
