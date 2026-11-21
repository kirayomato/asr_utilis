"""加载 + 静音转写，验证模型接入是否跑通。

用法（在仓库任意位置执行）：
    python .workbuddy/skills/asr-model-adapter/scripts/smoke_test.py "ark_audio_onnx('ark-asr-0.6b-int8-onnx')"
    python .../smoke_test.py "Qwen3ASRModel('Qwen3-ASR-0.6B', language=None)" --repo F:\\Program\\Github\\asr_utilis

表达式在 models 包的命名空间里求值，因此可以直接写类名并传参。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def build(expr: str, repo: Path):
    sys.path.insert(0, str(repo))
    import models

    namespace = {**vars(models), "models": models}
    return eval(expr, namespace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expr", help='形如 "ark_audio_onnx(\'ark-asr-0.6b-int8-onnx\')"')
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[4]),
        help="仓库根目录，默认按本脚本位置推断",
    )
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    model = build(args.expr, Path(args.repo).resolve())

    start = time.time()
    model.load()
    print(f"load: {time.time() - start:.1f}s")

    wav = np.zeros(int(16000 * args.seconds), dtype=np.float32)
    for _ in range(args.repeat):
        start = time.time()
        text = model.transcribe([wav])[0]
        print(f"text: {text!r} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
