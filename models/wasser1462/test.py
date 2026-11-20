from asr_model import Qwen3ASRModel
import librosa
# modelscope download jkman2023/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 --local_dir sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
model = Qwen3ASRModel(
    model_id="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    model_dir="Qwen3-ASR-0.6B-meta",
    context="微积分 微分几何 导数",
    language="Chinese",
)
model.load()
wav, sr = librosa.load(r"D:\Program\Github\Qwen3-ASR-onnx\test_wavs\rag_math.wav")
print(model.transcribe([wav]))
