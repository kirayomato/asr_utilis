import os
import numpy as np
import soundfile as sf
import torch
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
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
