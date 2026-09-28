"""下载 Qwen3 各尺寸基座模型（默认走 ModelScope，国内速度更快）。

用法：
    # 只下载 4B 和 8B
    python scripts/download_models.py --sizes 4b 8b

    # 权重放到别的盘
    python scripts/download_models.py --all --models-dir X:\models\cultist-llm

    # 走 HuggingFace
    python scripts/download_models.py --sizes 4b --source hf
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# 尺寸 → ModelScope / HuggingFace 上的模型 id
MODELS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
    "14b": "Qwen/Qwen3-14B",
}

# 各尺寸权重的粗略磁盘占用（GB），用于空间预检
APPROX_GB = {"0.6b": 1.5, "1.7b": 3.5, "4b": 8.0, "8b": 16.0, "14b": 28.0}

DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def local_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def free_gb(path: str) -> float:
    drive = os.path.splitdrive(os.path.abspath(path))[0] or os.path.abspath(path)
    return shutil.disk_usage(drive + os.sep).free / (1024 ** 3)


def download_modelscope(model_id: str, target: str) -> None:
    from modelscope import snapshot_download

    snapshot_download(model_id, local_dir=target)


def download_hf(model_id: str, target: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model_id, local_dir=target)


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 Qwen3 基座模型")
    ap.add_argument("--sizes", nargs="*", default=["4b"], choices=sorted(MODELS),
                    help="要下载的尺寸，可多选")
    ap.add_argument("--all", action="store_true", help="下载全部尺寸")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="模型存放根目录")
    ap.add_argument("--source", default="modelscope", choices=["modelscope", "hf"])
    ap.add_argument("--yes", action="store_true", help="跳过磁盘空间确认")
    args = ap.parse_args()

    sizes = sorted(MODELS) if args.all else args.sizes
    os.makedirs(args.models_dir, exist_ok=True)

    need = sum(APPROX_GB[s] for s in sizes)
    avail = free_gb(args.models_dir)
    print(f"目标目录：{args.models_dir}")
    print(f"待下载：{', '.join(sizes)}  约需 {need:.1f} GB，当前可用 {avail:.1f} GB")
    if need > avail:
        print(f"\n[错误] 磁盘空间不足（需要 {need:.1f} GB，可用 {avail:.1f} GB）。")
        print("       请换用 --models-dir 指向空间更大的盘，或减少 --sizes。")
        raise SystemExit(1)
    if avail - need < 5 and not args.yes:
        print(f"[提示] 下载后仅剩 {avail - need:.1f} GB，建议确认。加 --yes 可跳过此提示。")
        if input("继续？[y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit(1)

    downloader = download_modelscope if args.source == "modelscope" else download_hf

    for size in sizes:
        model_id = MODELS[size]
        target = os.path.join(args.models_dir, local_name(model_id))
        if os.path.isdir(target) and any(
            f.endswith((".safetensors", ".bin")) for f in os.listdir(target)
        ):
            print(f"[跳过] {target} 已存在")
            continue
        print(f"\n[下载] {model_id}  →  {target}")
        try:
            downloader(model_id, target)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {model_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("       可尝试换源：--source hf，或检查网络。")
            raise SystemExit(1)
        print(f"[完成] {model_id}")

    print("\n全部完成。模型目录：")
    for size in sizes:
        print("  ", os.path.join(args.models_dir, local_name(MODELS[size])))


if __name__ == "__main__":
    main()