import numpy as np
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

from .datasets import BaseASRDataset
from .metrics import cer_from_counts, count_edits
from .models import BaseASRModel
from .text import TextNormalizer


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

    # 计算评估指标
    print("计算评估指标...")
    counts = count_edits(normalized_targets, normalized_preds)
    cer = cer_from_counts(counts)

    # 准备结果
    results = {
        "model": model.model_id,
        "dataset": dataset.name,
        "samples": len(audio_data),
        "cer": cer,
        "cer_percentage": cer * 100,
        "hits": counts["hits"],
        "substitutions": counts["substitutions"],
        "deletions": counts["deletions"],
        "insertions": counts["insertions"],
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
            f.write(f"命中(H): {counts['hits']}\n")
            f.write(f"替换(S): {counts['substitutions']}\n")
            f.write(f"删除(D): {counts['deletions']}\n")
            f.write(f"插入(I): {counts['insertions']}\n")
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
