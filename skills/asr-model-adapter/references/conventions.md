# 接入约定与已知坑

## 1. 接口契约

`models/base.py`：

```python
class BaseASRModel(ABC):
    def __init__(self, model_id: str): ...   # 子类用 super().__init__(model_id) 存下来

    @abstractmethod
    def load(self): ...                      # 幂等；重复调用不应重复加载

    @abstractmethod
    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]: ...

    def batch_transcribe(self, audio_data, batch_size=32, progress_bar=True, loaded_data=None):
        # 自己会调 load()，把每批的音频用 np.pad 补零到批内最长，再调 transcribe()
```

要点：

- 音频输入是 **16kHz、单声道、float32（[-1,1]）**，可能带补零尾巴，`transcribe` 必须容忍长度不一致和纯静音。
- 返回 `List[str]`，顺序与输入一一对应；空音频返回空串，不要返回 None。
- `model_id` 有两种语义，二选一并在 docstring 里写明：**本地目录名**（走 `checkpoint_path`）或**上游库的模型 id**（如 `paraformer-zh`，由 funasr/modelscope 自己下载）。

## 2. 路径与下载

`models/paths.py`：

```python
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoint"

def checkpoint_path(name, repo_id=None) -> Path:
    # 1) 传进来的路径本身存在 -> 原样返回
    # 2) CHECKPOINT_DIR/<name> 存在 -> 返回
    # 3) 都没有且给了 repo_id -> modelscope.snapshot_download(repo_id, local_dir=...) 下载后返回
    # 4) 没给 repo_id -> FileNotFoundError，报出目标绝对路径
```

约定：

- 模型 py 里写 `REPO_ID = "owner/name"`，调用 `checkpoint_path(self.model_id, REPO_ID)`。
- 目录名保持下载时的原始名字，不做改名（`.mv` / `.msc` 里的相对路径依赖它）。
- 核对 repo id 是否存在：

```python
from modelscope.hub.api import HubApi
HubApi().get_model("Qwen/Qwen3-ASR-0.6B")   # 不存在会抛异常
```

- `snapshot_download(..., local_dir=...)` 会把文件直接放进目标目录，并附带 `.msc`、`.mv`、`._____temp` 三个标记；**只下载缺失模型，已有目录不联网**。
- 已知限制：`checkpoint_path` 只看目录是否存在。下载中断留下的半成品目录会被当成已就绪，需要人工删掉重下。

## 3. 目录布局差异

| 形态 | 处理 |
|---|---|
| 扁平包（onnx/tokenizer/config 全在包根） | 直接 `checkpoint_path` 当 root 用；若上游脚本假定子目录，见下 |
| 上游假定 `<root>/model` + `<root>/build` | 不改上游脚本，用 property 覆盖实例属性（见 `models/ark.py`） |
| 官方 safetensors + `transformers`/`qwen_asr` | `from_pretrained(str(checkpoint_path(...)))` |
| 只有 tokenizer / mel / meta 的配套目录 | 不算模型，放在使用它的包旁边（例：`models/wasser1462/Qwen3-ASR-0.6B-meta`），用 `Path(__file__).parent / "models" / ...` 定位 |

扁平包兜布局的写法（`models/ark.py` 的 `_build_runtime`）：

```python
class FlatLayoutRuntime(ArkAsrOnnxRuntime):
    def __init__(self, package_dir):
        self._package_dir = package_dir
        super().__init__(package_dir)          # 上游去读 model/runtime_manifest.json

    @property
    def model_dir(self):
        return self._package_dir               # 覆盖上游的 root/"model"

    @model_dir.setter
    def model_dir(self, value):
        pass                                   # 吞掉上游赋值

    # build_dir 同理
```

## 4. 已知模型清单

| 类 | 文件 | 权重位置 | ModelScope repo | 备注 |
|---|---|---|---|---|
| `ark_audio_onnx` | `models/ark.py` | `checkpoint/ark-asr-0.6b-int8-onnx` | `Audio8/ark-asr-0.6b-int8-onnx` | 扁平包 + 上游假定 model/build；临时 wav 写 temp |
| `Qwen3ASRModel` | `models/qwen3.py` | `checkpoint/Qwen3-ASR-0.6B` | `Qwen/Qwen3-ASR-0.6B` | 走 `qwen_asr`，bf16 + cuda:0 |
| `Qwen3ASRModel` | `models/andrewleech/asr_model.py` | `checkpoint/qwen3-asr-0.6b-onnx-int4-merged` | `rhasspy/qwen3-asr-0.6b-onnx-int4-merged` | 第三方包自带 ONNX 推理 |
| `Qwen3ASRModel` | `models/wasser1462/asr_model.py` | `checkpoint/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` + `models/wasser1462/Qwen3-ASR-0.6B-meta` | `jkman2023/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | 同上，meta 是配套目录不是模型 |
| `GLMASRLocalModel` | `models/glm_local.py` | `D:\Program\VLM\cache\ZhipuAI\GLM-ASR-Nano-2512` | 未纳入 checkpoint | 4bit 量化，权重在仓库外 |
| `FunASRModel` / `paraformer_onnx` | `models/funasr_model.py` | 上游库自管 | `paraformer-zh`、`iic/SenseVoiceSmall` 等 | 远端 id，非本地路径 |
| `WhisperModel` / `Wav2Vec2Model` / `Nemotron` / `x_asr` / `zipformer` | 各自文件 | 上游库自管 | HF / sherpa | 非本地路径 |

## 5. 踩过的坑

- **不要在 `load()` 里 `os.chdir(model_dir)`**。会污染全局 cwd，`realtime_speech.generate()` 的 `os.makedirs("transcribe")` 和转写 txt 会写到模型目录里去。
- **不要写死盘符绝对路径**。换机器/换盘就炸；一律走 `checkpoint_path`。
- **临时音频写系统 temp**（`models/ark.py` 的 `WAV_PATH = Path(tempfile.gettempdir()) / ...`），别在仓库根目录留 `soundfile.wav`。
- **tokenizer 元数据不是模型**，不要塞进 `models/checkpoint`。
- **半成品目录**：下载中断后目录还在但文件不全，`checkpoint_path` 会当作就绪。重下要手动删目录。
- **第三方目录保持只读**：`infer_ark_audio_onnx.py` 之类被我们 patch 过一次就不好升级，兼容层写在 wrapper 里。
- `need_punc=True` 时 `realtime_speech` 会额外加载 `funasr_onnx` 的 `CT_Transformer`（模型 id `iic/punc_ct-transformer_...`），首次跑会联网下载；不需要标点就传 `need_punc=False`。
- `realtime_speech.record()` 依赖 pyaudio 和虚拟声卡设备名（`Line {n} (Virtual Audio Cable)`）。没有设备时用 `scripts/smoke_test.py` 验证，不要直接跑入口脚本。
