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
        "steps": 6720,
        "batch": "4 × 4",
        "runtime": "1 小时 22 分",
        "eval_loss": "2.406",
        "merged_size": "1.13 GB",
        "lora_size": "38.5 MB",
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
    },
    "4b": {
        "base": "Qwen/Qwen3-4B",
        "lora_dir": "outputs/qwen3-4b-lora",
        "merged_dir": "outputs/qwen3-4b-lora-merged",
        "params": "4B",
        "epochs": 3,
        "steps": 5040,
        "batch": "1 × 16",
        "runtime": "6 小时",
        "eval_loss": "1.949",
        "merged_size": "7.51 GB",
        "lora_size": "126.1 MB",
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

CARD = """---
library_name: transformers
base_model: {base}
base_model_relation: finetune
language:
  - zh
  - en
license: apache-2.0
pipeline_tag: text-generation
tags:
  - qwen3
  - cultist-simulator
  - game-text
  - lora
  - peft
  - text-generation
  - chinese
  - english
---

# cultist-simulator-qwen3-{size}

用《密教模拟器》(Cultist Simulator) 的游戏内文本微调 [{base}](https://huggingface.co/{base}) 得到的 {params} 文本生成模型。目标是让它学会这个游戏那种晦涩、克制、带书卷气又带点不祥暗示的写法，能写卡牌描述、事件叙事、结局文本，也能做中英互译。生成的文字可以直接拿去当同人创作、跑团素材或模组草稿。

## 模型详情

基座是 [{base}](https://huggingface.co/{base})，参数量 {params}，用 LoRA 微调（`r=16` / `alpha=32`，不是 QLoRA，权重为 bf16），训练框架是 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 0.9.4。支持中文和英文，许可跟随基座为 Apache-2.0。

仓库根目录放的是合并后的完整模型，LoRA 已经并入权重，`from_pretrained` 直接可加载，大小 {merged_size}；`lora/` 子目录放 LoRA 适配器（`r=16`），配合基座模型使用，大小 {lora_size}。按需要取其中一个就行。

## 训练细节

| 项目 | 值 |
| --- | --- |
| 基座模型 | [{base}](https://huggingface.co/{base}) |
| 微调方式 | LoRA（`r=16`，`alpha=32`，作用于 q/k/v/o/gate/up/down 全部投影层） |
| 训练样本 | 26,875 条（验证集 779 条，按实体 id 稳定划分） |
| 训练轮数 | {epochs} epoch / {steps} 步 |
| 等效 batch | {batch}（`per_device × 梯度累积` = 16） |
| 截断长度 | 896 |
| 学习率 | 1e-4（cosine，warmup 10%） |
| 精度 | bf16 + 梯度检查点 |
| 对话模板 | `qwen3_nothink` |
| 最终 eval_loss | **{eval_loss}** |
| 训练耗时 | {runtime} |
| 硬件 | 单卡 16GB 显存（bf16） |

数据全部来自本地已安装的游戏本体，不依赖任何联网数据源。抽取与构建脚本见文末。

### 训练了什么任务

数据集是多任务的，同一条语料会派生出多个样本：

| 任务 | 说明 |
| --- | --- |
| 中英互译 | 两侧都带 `<b>`/`<i>` 标记时才要求保留，否则把标记一并剥掉，保证指令与答案自洽 |
| 卡牌描述 | 给定卡牌名 + 类别 + 主题，写描述文本 |
| 行动叙事 | 给定行动类型 + 事件名，写开场叙述 / 结算叙述 |
| 结局正文 | 按结局名撰写正文 |
| 主题创作 | 以准则（灯/铸/刃/冬/心/杯/蛾/启/秘史）为主题创作 |
| 风格续写 | 给定开头，按原文风格续写 |

## 快速使用

### 完整模型（推荐，最省事）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

system = (
    "你是《密教模拟器》（Cultist Simulator）的文本生成模型。"
    "你精通该游戏晦涩、克制、带着书卷气与不祥暗示的文风，"
    "能够按给定的题材、类别和主题创作中文与英文游戏文本，也能在中英文之间准确互译。"
)
user = "为《密教模拟器》中的一张卡牌撰写描述文本。\\n\\n语言：中文\\n卡牌名称：无光的陈列室\\n类别：地点\\n主题：冬、灯"

msgs = [{{"role": "system", "content": system}}, {{"role": "user", "content": user}}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = model.generate(**tok(text, return_tensors="pt").to(model.device), max_new_tokens=256)
print(tok.decode(out[0][tok(text)["input_ids"].__len__():], skip_special_tokens=True))
```

### LoRA 适配器

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base}", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="lora")
```

### Ollama

仓库根目录带 LLaMA-Factory 生成的 `Modelfile`，下载后在本目录下执行：

```bash
ollama create cultist-{params} -f Modelfile
ollama run cultist-{params}
```

## 效果示例

同一组提示词下，4B 与 0.6B 的对比（4B 全面更好）：

**卡牌描述**（输入：`卡牌名称：无光的陈列室 / 类别：地点 / 主题：冬、灯`）

- 4B：`雪覆一切，日沉至地平线下。光线如冰霜般冷冽。`
- 0.6B：`这里的每一块石头都深陷青苔，而石缝里那盆枯萎的多肉植物的根茎仍在挣扎。`

**中译英（要求保留 `<b>` 标记）**（输入：`光透过<b>裂缝漏入</b>。我的头脑比<b>任何时候</b>都<b>更清晰</b>。`）

- 4B：`The light leaks through <b>cracks</b>. My mind is <b>clearer</b> than it's been <b>in any other time</b>.` ✅ 标记完整保留
- 0.6B：`Light leaks through a Crack in the Skin. My brain is clearer than ever at any given time.` ❌ 标记全丢

> eval_loss 也印证这个差距：4B 1.949 vs 0.6B 2.406。

## 数据来源与使用限制

训练语料全部抽取自《密教模拟器》游戏本体，版权归 Weather Factory 所有，中文文本来自游戏的官方简体中文资源。基座模型 {base} 采用 Apache-2.0 协议，本仓库的权重是它的微调衍生品。

请只把本模型用于个人学习、同人创作这类非商业用途。模型生成的内容有可能重现或近似游戏原文，公开发布生成结果时请遵守游戏原作者的授权条款并注明题材来源。另外它只学了文风和文本结构，并不理解也不会执行游戏机制，生成内容不保证与官方设定一致。

## 已知局限

训练数据只含叙事性文本字段，模型不知道卡牌的实际效果、配方条件与数值。因为训练集来自游戏本体，它有一定概率逐字输出或近似改写官方文本，直接用在作品里之前请自行确认授权。0.6B 这种小尺寸在需要严格保留格式的任务上会力不从心，比如中译英保留 `<b>` 标记就会丢，对格式遵循度要求高的场景建议用 4B。语言上只覆盖简体中文和英文，其他语言没训过。输出长度还受训练时 `cutoff_len=896` 影响，明显更长的连续文本不是它的强项。

## 复现

完整的数据抽取、训练、推理与发布流程见 GitHub 仓库：

```bash
git clone https://github.com/{github}.git
cd cultist-llm

# 抽取游戏文本 → 构建多任务数据集（需要本地已安装游戏本体）
.\\scripts\\prepare_data.ps1

# 训练 + 合并
.\\scripts\\train.ps1 -Size {size} -Merge
```

## 同系列模型

| 尺寸 | 仓库 |
| --- | --- |
| 0.6B | [`{ns}/cultist-simulator-qwen3-0.6b`](https://huggingface.co/{ns}/cultist-simulator-qwen3-0.6b) |
| 4B | [`{ns}/cultist-simulator-qwen3-4b`](https://huggingface.co/{ns}/cultist-simulator-qwen3-4b) |

## 相关链接

- 源码与训练脚本：<https://github.com/{github}>
- 基座模型：<https://huggingface.co/{base}>
"""


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
                print("    README.md（模型卡，模板生成）")
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
                path_or_fileobj=CARD.format(
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
