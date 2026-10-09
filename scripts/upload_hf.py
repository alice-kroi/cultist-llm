"""把微调好的《密教模拟器》模型上传到 Hugging Face Hub。

每个尺寸一个仓库：

    <repo>/
        <合并后的完整模型>   # 根目录，from_pretrained 直接加载
        lora/                # LoRA 适配器，配合基座模型使用
        README.md            # 模型卡（本脚本生成）

合并模型放根目录、适配器放 lora/，这样两种用法都不歧义：
from_pretrained(repo) 拿完整模型，
PeftModel.from_pretrained(base, repo, subfolder="lora") 拿适配器。

用法：
    # 干跑，只列清单不碰网络
    python scripts/upload_hf.py --sizes 0.6b,4b --token hf_xxx --dry-run

    # 实际上传（默认公开仓库）
    python scripts/upload_hf.py --sizes 0.6b,4b --token hf_xxx

    python scripts/upload_hf.py --sizes 4b --token hf_xxx --skip-merged
    python scripts/upload_hf.py --sizes 4b --token hf_xxx --skip-lora
    python scripts/upload_hf.py --sizes 4b --token hf_xxx --private
    python scripts/upload_hf.py --sizes 4b --token hf_xxx --repo-id me/my-model

    # 只更新模型卡，不必重传几 GB 权重
    python scripts/upload_hf.py --sizes 0.6b,4b --token hf_xxx --card-only

关于 token：HF_TOKEN 环境变量的优先级高于显式传入的 token=，所以给了 --token
之后脚本会先把它从环境里摘掉。另外用 SetEnvironmentVariable 改的环境变量只写
注册表，已经开着的进程还是旧值，要新开终端才生效。
"""

from __future__ import annotations

import argparse
import os
import sys

# hf_transfer 走 Rust 的多线程分片上传，大文件快不少；但直连不通要走代理时反而
# 得关掉它（见 README 的「网络不通时的代理设置」）。
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# 端点在 import huggingface_hub 之前就得设好：constants.ENDPOINT 是导入时求值的，
# 之后再改 os.environ 已经晚了（HfApi 也只有一个不带 endpoint 的构造）。
# 如果 HF_ENDPOINT 被设成失效的镜像，那种网络故障会被告成 `Invalid user token`，
# 看着像令牌坏了其实是网络；这里的预扫描就是为了能临时绕开。
for _i, _a in enumerate(sys.argv):
    if _a == "--endpoint" and _i + 1 < len(sys.argv):
        os.environ["HF_ENDPOINT"] = sys.argv[_i + 1]
        break

from huggingface_hub import HfApi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# 每个尺寸的产物目录、基座模型、训练超参与实测结果（数字都来自实际训练日志）
SIZES = {
    "0.6b": {
        "base": "Qwen/Qwen3-0.6B",
        "lora_dir": "outputs/qwen3-0.6b-lora",
        "merged_dir": "outputs/qwen3-0.6b-lora-merged",
        "params": "0.6B",
        "epochs": 4,
        "steps": 13672,
        "batch": "4 × 4",
        "runtime": "7 小时 54 分",
        "eval_loss": "2.070",
        "merged_size": "1.13 GB",
        "lora_size": "38.5 MB",
        "train_samples": "54,683",
        "eval_samples": "6,210",
    },
    "1.7b": {
        "base": "Qwen/Qwen3-1.7B",
        "lora_dir": "outputs/qwen3-1.7b-lora",
        "merged_dir": "outputs/qwen3-1.7b-lora-merged",
        "params": "1.7B",
        "epochs": 3,
        "steps": 5040,
        "batch": "2 × 8",
        "runtime": "—",
        "eval_loss": "—",
        "merged_size": "—",
        "lora_size": "—",
        "train_samples": "—",
        "eval_samples": "—",
    },
    "4b": {
        "base": "Qwen/Qwen3-4B",
        "lora_dir": "outputs/qwen3-4b-lora",
        "merged_dir": "outputs/qwen3-4b-lora-merged",
        "params": "4B",
        "epochs": 3,
        "steps": 11067,
        "batch": "1 × 16",
        "runtime": "15 小时 41 分",
        "eval_loss": "1.633",
        "merged_size": "7.51 GB",
        "lora_size": "126.1 MB",
        "train_samples": "59,018",
        "eval_samples": "1,858",
    },
    "8b": {
        "base": "Qwen/Qwen3-8B",
        "lora_dir": "outputs/qwen3-8b-qlora",
        "merged_dir": "outputs/qwen3-8b-qlora-merged",
        "params": "8B",
        "epochs": 3,
        "steps": 5040,
        "batch": "1 × 16",
        "runtime": "—",
        "eval_loss": "—",
        "merged_size": "—",
        "lora_size": "—",
        "train_samples": "—",
        "eval_samples": "—",
    },
    "14b": {
        "base": "Qwen/Qwen3-14B",
        "lora_dir": "outputs/qwen3-14b-qlora",
        "merged_dir": "outputs/qwen3-14b-qlora-merged",
        "params": "14B",
        "epochs": 3,
        "steps": 5040,
        "batch": "1 × 16",
        "runtime": "—",
        "eval_loss": "—",
        "merged_size": "—",
        "lora_size": "—",
        "train_samples": "—",
        "eval_samples": "—",
    },
}

# LoRA 目录里只挑这些传；checkpoint-*、trainer_state.json、training_args.bin
# 这些中间产物对使用者没意义，不传。
LORA_KEEP = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

# 模型卡里回链到 GitHub 源码仓库用。改仓库地址只改这一处。
GITHUB_REPO = "alice-kroi/cultist-llm"

# 模型卡的 HF 元数据与正文都在这个模板里，直接编辑它即可，不必改代码
CARD_TEMPLATE = os.path.join(PROJECT_ROOT, "templates", "model_card.md")


def human(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.2f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.2f} GB"


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def collect(spec: dict, skips: tuple[str, ...]) -> list[str]:
    """列出将要上传的本地文件（相对路径）。"""
    out = []
    lora = os.path.join(PROJECT_ROOT, spec["lora_dir"])
    merged = os.path.join(PROJECT_ROOT, spec["merged_dir"])
    if "merged" not in skips:
        if os.path.isdir(merged):
            for f in sorted(os.listdir(merged)):
                if os.path.isfile(os.path.join(merged, f)):
                    out.append(f"[root] {f}")
        else:
            out.append(f"[root] !! 目录不存在：{merged}")
    if "lora" not in skips:
        if os.path.isdir(lora):
            for f in LORA_KEEP:
                p = os.path.join(lora, f)
                if os.path.isfile(p):
                    out.append(f"[lora] {f}")
        else:
            out.append(f"[lora] !! 目录不存在：{lora}")
    return out


def load_card() -> str:
    """读取模型卡模板。每次上传都重新读，改完模板不必重启。"""
    if not os.path.isfile(CARD_TEMPLATE):
        raise SystemExit(f"模型卡模板不存在：{CARD_TEMPLATE}")
    with open(CARD_TEMPLATE, encoding="utf-8") as fh:
        return fh.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="上传《密教模拟器》微调模型到 Hugging Face")
    ap.add_argument("--sizes", default="0.6b,4b", help="逗号分隔，可选：0.6b/1.7b/4b/8b/14b")
    ap.add_argument("--token", default="", help="Hugging Face access token（write 权限）")
    ap.add_argument(
        "--endpoint",
        default="",
        help="HF 端点，如 https://huggingface.co；不给则沿用 HF_ENDPOINT 环境变量",
    )
    ap.add_argument("--repo-id", default="", help="完整仓库名 user/name；只传单个尺寸时可用")
    ap.add_argument("--namespace", default="", help="仓库归属用户名，默认用 token 对应的账号")
    ap.add_argument("--private", action="store_true", help="建私有仓库（默认公开）")
    ap.add_argument("--skip-lora", action="store_true", help="不传 LoRA 适配器")
    ap.add_argument("--skip-merged", action="store_true", help="不传合并后的完整模型")
    ap.add_argument("--card-only", action="store_true", help="只更新模型卡 README.md，不传权重")
    ap.add_argument("--dry-run", action="store_true", help="只列出要传的内容，不碰网络")
    args = ap.parse_args()

    sizes = [s.strip().lower() for s in args.sizes.split(",") if s.strip()]
    unknown = [s for s in sizes if s not in SIZES]
    if unknown:
        print(f"[错误] 未知尺寸：{', '.join(unknown)}（可选：{', '.join(SIZES)}）")
        return 1
    if args.card_only:
        # 只改模型卡文案时不必重传几 GB 权重
        skips = ("lora", "merged")
    elif args.skip_lora and args.skip_merged:
        print("[错误] --skip-lora 和 --skip-merged 不能同时给（只想更新模型卡请用 --card-only）")
        return 1
    else:
        skips = tuple(
            k for k, v in (("lora", args.skip_lora), ("merged", args.skip_merged)) if v
        )
    if args.repo_id and len(sizes) != 1:
        print("[错误] --repo-id 只能配单个尺寸使用")
        return 1

    # ---------- 干跑：只列清单 ----------
    if args.dry_run:
        for size in sizes:
            spec = SIZES[size]
            print(f"\n=== {size}  ({spec['base']}) ===")
            if args.card_only:
                # 模型卡不在 collect() 的结果里，不点明的话这里会是一片空白
                print("    README.md（模型卡，由 templates/model_card.md 生成）")
            for line in collect(spec, skips):
                print("   ", line)
            # 只统计真正会上传的文件：merged 传整个目录，lora 只传 LORA_KEEP 里那几个，
            # 别把 checkpoint-* 算进去，否则数字会大出一大截
            if "merged" not in skips:
                path = os.path.join(PROJECT_ROOT, spec["merged_dir"])
                if os.path.isdir(path):
                    total = sum(
                        os.path.getsize(os.path.join(path, f))
                        for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))
                    )
                    print(f"    → 合并模型合计 {human(total)}")
            if "lora" not in skips:
                path = os.path.join(PROJECT_ROOT, spec["lora_dir"])
                if os.path.isdir(path):
                    total = sum(
                        os.path.getsize(os.path.join(path, f))
                        for f in LORA_KEEP
                        if os.path.isfile(os.path.join(path, f))
                    )
                    print(f"    → LoRA 合计 {human(total)}（仅适配器与分词器，不含 checkpoint）")
        print("\n[dry-run] 未发起任何网络请求。去掉 --dry-run 即开始上传。")
        return 0

    # ---------- 真实上传 ----------
    if not args.token:
        print("[错误] 需要 --token。")
        print("       在 https://huggingface.co/settings/tokens 建一个 **Write** 权限的令牌。")
        print("       注意：不要依赖 HF_TOKEN 环境变量——它的优先级最高，会顶掉 --token。")
        return 1

    if args.token:
        # HF_TOKEN 环境变量的优先级高于显式传入的 token=，不摘掉的话 --token 会
        # 静默失效：HfApi(token=新令牌).whoami() 依然报 Invalid user token。
        os.environ.pop("HF_TOKEN", None)

    api = HfApi(token=args.token)
    from huggingface_hub import constants as _const

    print(f"[端点] {_const.ENDPOINT}")
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] token 校验失败：{exc}")
        print("       两种常见原因：")
        print("       1) 令牌无效或没有 write 权限；")
        print(f"       2) 端点不通。当前端点 {_const.ENDPOINT}，若不是 https://huggingface.co，")
        print("          加 --endpoint https://huggingface.co 再试（镜像故障会被误报成令牌无效）。")
        return 1
    namespace = args.namespace or who.get("name")
    print(f"[认证] 已登录为 {namespace}")

    failed = []
    for size in sizes:
        spec = SIZES[size]
        repo_id = args.repo_id or f"{namespace}/cultist-simulator-qwen3-{size}"
        print(f"\n{'=' * 62}\n=== {size} → {repo_id} ===\n{'=' * 62}")
        print("".join(f"    {line}\n" for line in collect(spec, skips)).rstrip())

        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
            print(f"[仓库] 已就绪：https://huggingface.co/{repo_id}")

            # 先传模型卡，仓库立刻有内容；再传权重
            api.upload_file(
                path_or_fileobj=load_card().format(
                    size=size,
                    params=spec["params"],
                    base=spec["base"],
                    repo=repo_id,
                    ns=namespace,
                    github=GITHUB_REPO,
                    epochs=spec["epochs"],
                    steps=spec["steps"],
                    batch=spec["batch"],
                    runtime=spec["runtime"],
                    eval_loss=spec["eval_loss"],
                    merged_size=spec["merged_size"],
                    lora_size=spec["lora_size"],
                    train_samples=spec["train_samples"],
                    eval_samples=spec["eval_samples"],
                ).encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
            )
            print("[上传] README.md（模型卡）完成")

            if "merged" not in skips:
                merged = os.path.join(PROJECT_ROOT, spec["merged_dir"])
                print(f"[上传] 合并模型根目录（{human(dir_size(merged))}），大文件较慢…")
                api.upload_folder(repo_id=repo_id, folder_path=merged, path_in_repo=".")
                print("[上传] 合并模型完成")

            if "lora" not in skips:
                lora = os.path.join(PROJECT_ROOT, spec["lora_dir"])
                print("[上传] LoRA 适配器 → lora/")
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=lora,
                    path_in_repo="lora",
                    allow_patterns=LORA_KEEP,
                )
                print("[上传] LoRA 适配器完成")

            print(f"[完成] https://huggingface.co/{repo_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {size}：{exc}")
            failed.append(size)

    if failed:
        print(f"\n以下尺寸上传失败：{', '.join(failed)}")
        return 1
    print("\n全部上传完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
