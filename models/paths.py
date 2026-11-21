from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoint"


def checkpoint_path(name: str | Path, repo_id: str | None = None) -> Path:
    """本地模型统一放在 models/checkpoint 下，缺失时用 repo_id 从 ModelScope 下载

    传目录名即可；传入已存在的路径则原样返回。
    """
    path = Path(name).expanduser()
    if path.is_dir():
        return path

    target = CHECKPOINT_DIR / path
    if target.is_dir():
        return target

    if not repo_id:
        raise FileNotFoundError(f"模型目录不存在，且未提供 ModelScope 仓库 id: {target}")
    _download(repo_id, target)
    return target


def _download(repo_id: str, target: Path) -> None:
    from modelscope import snapshot_download

    print(f"本地缺少 {target.name}，从 ModelScope 下载 {repo_id} -> {target}", flush=True)
    snapshot_download(repo_id, local_dir=str(target))
