from threading import Thread
import numpy as np
import torch
from realtime_speech import start_transcribe
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
    BitsAndBytesConfig,
)

WHISPER_FEAT_CFG = {
    "chunk_length": 10,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}


def load_model(checkpoint_dir: str, tokenizer_path: str | None, device: str) -> tuple:
    """
    加载ASR模型和相关组件

    Args:
        checkpoint_dir: 模型检查点目录路径
        tokenizer_path: 分词器路径（可选）
        device: 设备类型 ('cuda' 或 'cpu')

    Returns:
        tuple: (model, tokenizer, feature_extractor, config)
    """
    tokenizer_source = tokenizer_path if tokenizer_path else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16,  # 改为float16以避免类型冲突
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()

    return model, tokenizer, feature_extractor, config


def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1

    # 限制单次30秒chunk的最大token数
    audio_token_num = min(audio_token_num, 1500)
    return audio_token_num


def prepare_single_chunk_input(
    chunk_wav,
    tokenizer,
    feature_extractor,
    merge_factor: int,
):
    """
    处理单个音频片段(30s)，构建 Prompt
    """
    # 提取特征
    mel = feature_extractor(
        chunk_wav.numpy(),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    )["input_features"]

    # 计算实际秒数
    seconds = chunk_wav.shape[1] / feature_extractor.sampling_rate
    num_tokens = get_audio_token_length(seconds, merge_factor)

    # 构建 Prompt Token
    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")

    # 音频部分
    tokens += tokenizer.encode("<|begin_of_audio|>")
    audio_offset = len(tokens)
    tokens += [0] * num_tokens
    tokens += tokenizer.encode("<|end_of_audio|>")
    audio_len = num_tokens

    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")
    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": mel,  # [1, 128, 3000]
        "audio_offsets": [[audio_offset]],
        "audio_length": [[audio_len]],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch


def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
    tokens = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audios = batch["audios"].to(
        device, dtype=torch.float16
    )  # 确保音频数据类型为float16
    model_inputs = {
        "inputs": tokens,
        "attention_mask": attention_mask,
        "audios": audios,
        "audio_offsets": batch["audio_offsets"],
        "audio_length": batch["audio_length"],
    }
    return model_inputs, tokens.size(1)


def _transcribe(
    audio_chunk: np.ndarray,
    model,
    tokenizer,
    feature_extractor,
    config,
    max_new_tokens: int = 128,
):
    """
    转录单个音频片段
    """
    # 转换为tensor并确保正确形状
    if len(audio_chunk.shape) == 1:
        audio_chunk = audio_chunk.reshape(1, -1)

    chunk_wav = torch.from_numpy(audio_chunk).to(
        dtype=torch.float16
    )  # 确保输入数据类型为float16

    # 准备输入
    batch = prepare_single_chunk_input(
        chunk_wav,
        tokenizer,
        feature_extractor,
        merge_factor=config.merge_factor,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inputs, prompt_len = prepare_inputs(batch, device)

    # 推理
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
        )

    # 解码结果
    transcript_ids = generated[0, prompt_len:].cpu().tolist()
    text_chunk = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()

    return text_chunk


model, tokenizer, feature_extractor, config = load_model(
    checkpoint_dir=r"D:\Program\VLM\cache\ZhipuAI\GLM-ASR-Nano-2512",
    tokenizer_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def transcribe(wav: np.ndarray) -> str:
    return _transcribe(
        wav,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=config,
        max_new_tokens=128,
    )


start_transcribe(transcribe, False)
