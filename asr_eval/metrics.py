import jiwer
from typing import Dict, List


def count_edits(references: List[str], hypotheses: List[str]) -> Dict[str, int]:
    """逐句累计命中/替换/删除/插入计数

    jiwer 按空格切分后对齐，粒度为调用方传入文本的词。中文经
    TextNormalizer 拆成单字后，这里的"词"即单字，统计结果等价于字级。
    """
    hits = substitutions = deletions = insertions = 0

    for reference, hypothesis in zip(references, hypotheses):
        try:
            chunk_metrics = jiwer.process_words(reference, hypothesis)
            hits += chunk_metrics.hits
            substitutions += chunk_metrics.substitutions
            deletions += chunk_metrics.deletions
            insertions += chunk_metrics.insertions
        except Exception as e:
            print(f"处理错误: {e}\n源文本: {reference}\n目标文本: {hypothesis}")

    return {
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def cer_from_counts(counts: Dict[str, int]) -> float:
    """由 H/S/D/I 计数算 (S+D+I)/(H+S+D)，无有效字时返回 0"""
    denominator = counts["hits"] + counts["substitutions"] + counts["deletions"]
    if denominator <= 0:
        return 0.0
    return (
        counts["substitutions"] + counts["deletions"] + counts["insertions"]
    ) / denominator
