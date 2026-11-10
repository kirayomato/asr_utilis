"""模型封装，每个模型族一个模块，对外统一 BaseASRModel 接口"""

from .ark import ark_audio_onnx
from .base import BaseASRModel
from .funasr_model import FunASRModel, paraformer_onnx
from .glm import GLMASRLLMModel
from .glm_local import GLMASRLocalModel
from .nemotron import Nemotron
from .qwen3 import Qwen3ASRModel
from .sherpa import x_asr, zipformer
from .wav2vec2 import Wav2Vec2Model
from .whisper import WhisperModel

__all__ = [
    "BaseASRModel",
    "FunASRModel",
    "WhisperModel",
    "Wav2Vec2Model",
    "paraformer_onnx",
    "zipformer",
    "x_asr",
    "Qwen3ASRModel",
    "Nemotron",
    "ark_audio_onnx",
    "GLMASRLLMModel",
    "GLMASRLocalModel",
]
