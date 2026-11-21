---
name: asr-model-adapter
description: 在 asr_utilis 仓库接入新的 ASR 模型（本地 checkpoint、ONNX 包、HuggingFace/ModelScope 模型）时使用；也用于把已有模型改成 models/checkpoint 统一布局、补 ModelScope 自动下载、排查模型加载时的路径或目录布局报错。触发词：接入模型、适配模型、新增 ASR、换模型、模型路径、checkpoint、modelscope 下载、runtime_manifest.json、FileNotFoundError 模型。
agent_created: true
---

# ASR 模型适配

## 目标

把任意 ASR 模型接成仓库统一的 `BaseASRModel` 子类：权重放在 `models/checkpoint`，缺了自动从 ModelScope 拉，入口脚本能直接 `start_transcribe` 跑起来。

## 三条硬约定

1. **权重只在 `models/checkpoint/<目录名>`**，取路径一律 `checkpoint_path(name, REPO_ID)`。禁止写绝对路径（如 `F:\...\models\xxx`）、禁止 `os.chdir` 切到模型目录、禁止依赖“从仓库根启动”。
2. **`REPO_ID` 写在模型自己的 py 文件里**（模块级常量）。第三方模型包 `models/andrewleech`、`models/wasser1462` 不改源码，repo id 写在调用它的入口脚本里。
3. **不改下载下来的第三方目录里的脚本**（如 `infer_ark_audio_onnx.py`）。上游对目录布局的假设，在自己的 wrapper 里兜住。

## 接入流程

1. **确认模型形态**，决定落点：
   - 官方权重（safetensors / ONNX 包）：`models/checkpoint/<目录名>`，目录名保持仓库/包的原始名字。
   - 只有 tokenizer、mel、meta 之类的**配套文件不算模型**，跟随它所属的包存放（例：`models/wasser1462/Qwen3-ASR-0.6B-meta`）。
2. **查仓库 id**：ModelScope 上确认 `owner/name` 确实存在，再写进常量。核对方式见 `references/conventions.md`。
3. **写 wrapper**：复制 `assets/model_template.py` 到 `models/<模型名>.py`，实现 `load()` 与 `transcribe()`。ONNX/自包含包的推理代码若已由上游提供，import 上游脚本 + 兜布局，参考 `models/ark.py`。
4. **导出**：在 `models/__init__.py` 里 import 并加进 `__all__`。
5. **写入口**：新建 `real_offline_<模型名>.py`，调 `start_transcribe(model, need_punc=..., timing=...)`。
6. **自检**：`python .workbuddy/skills/asr-model-adapter/scripts/smoke_test.py "ark_audio_onnx('ark-asr-0.6b-int8-onnx')"`，加载 + 2s 静音转写要能出结果。
7. **回归**：至少再跑一个已有模型（如 `tmp/test_ark_load.py`），确认公共路径没被改坏。

## 自检清单

- [ ] 模型目录在 `models/checkpoint` 下，代码里没有任何盘符绝对路径
- [ ] 目录缺失时 `checkpoint_path(name, REPO_ID)` 能自动下载（模型 py 里有 `REPO_ID`）
- [ ] `load()` 内没有 `os.chdir`，临时文件写系统 temp
- [ ] `models/__init__.py` 已导出，`from models import <类名>` 可用
- [ ] 在仓库外目录执行入口脚本/自检脚本同样正常（cwd 无关）
- [ ] `transcribe()` 能接受 `batch_transcribe` 补零对齐后的批数据（16kHz、单声道 float32）
- [ ] 入口脚本跑起来有输出，并写出 `transcribe/transcribe_<时间>.txt`

## 参考

- `references/conventions.md`：接口签名、`batch_transcribe` 行为、目录布局差异（扁平包 / root+model+build / 半成品目录）、已知模型清单、踩过的坑。
- `assets/model_template.py`：可直接复制的最小 wrapper 模板。
- `scripts/smoke_test.py`：加载 + 静音转写的通用自检脚本。
