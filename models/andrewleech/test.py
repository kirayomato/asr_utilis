from asr_model import Qwen3ASRModel
import librosa

# modelscope download rhasspy/qwen3-asr-0.6b-onnx-int4-merged

m = Qwen3ASRModel(
        "qwen3-asr-0.6b-onnx-int4-merged",
        context="词汇: 微分几何,积分",
        language="Chinese",
    )
m.load()
wav, _ = librosa.load(r"D:\Program\Github\Qwen3-ASR-onnx\test_wavs\rag_math.wav")
print(m.transcribe([wav]))  # 应与 split 版逐字相同
