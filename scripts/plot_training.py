"""把 LLaMA-Factory 训练产物画成指标图表，用于训练后的效果分析与后续开发。

数据来源（都在训练输出目录 output_dir 下）：
  trainer_log.jsonl   每 logging_steps 一行：loss / lr / epoch / elapsed_time
  trainer_state.json  log_history：每步的 grad_norm / learning_rate / loss，
                      以及 eval 时的 eval_loss / 吞吐
  all_results.json    训练结束后的最终指标（train_loss / eval_loss / 吞吐 / 耗时）

产出：
  <输出>/training_metrics.png  多面板图：训练 loss、验证 loss、学习率、梯度范数、每步耗时

用法：
    python scripts/plot_training.py --output-dir outputs\\qwen3-4b-lora
    python scripts/plot_training.py --output-dir outputs\\qwen3-4b-lora --out logs\\metrics_4b.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 优先用中文字体，避免标题里的中文变成方块
for _name in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
    if any(f.name == _name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name]
        break
plt.rcParams["axes.unicode_minus"] = False


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_json(path: str) -> dict | None:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="LLaMA-Factory 训练指标图表")
    ap.add_argument("--output-dir", required=True, help="训练输出目录（含 trainer_state.json 等）")
    ap.add_argument("--out", default="", help="图表保存路径；默认写到仓库 logs/ 下 metrics_<name>.png")
    args = ap.parse_args()

    out_dir = args.output_dir
    if not os.path.isdir(out_dir):
        print(f"[错误] 输出目录不存在：{out_dir}")
        return 1

    state = load_json(os.path.join(out_dir, "trainer_state.json")) or {}
    logs = load_jsonl(os.path.join(out_dir, "trainer_log.jsonl"))
    results = load_json(os.path.join(out_dir, "all_results.json")) or {}

    hist = state.get("log_history") or []
    if not hist:
        print(f"[错误] {out_dir} 里没有可用的训练记录（trainer_state.json 为空），可能训练没跑完。")
        return 1

    # ---- 整理曲线 ----
    steps, losses, lrs, grad_norms = [], [], [], []
    eval_steps, eval_losses = [], []
    for entry in hist:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            steps.append(step)
            losses.append(entry["loss"])
        if "learning_rate" in entry:
            lrs.append(entry["learning_rate"])
        if "grad_norm" in entry:
            grad_norms.append((step, entry["grad_norm"]))
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

    # 每步耗时：trainer_log.jsonl 里有累计 elapsed_time，逐条差分即每段耗时
    per_step = []
    if len(logs) >= 2:
        for prev, cur in zip(logs[:-1], logs[1:]):
            t0, t1 = prev.get("elapsed_time"), cur.get("elapsed_time")
            if isinstance(t0, str) and isinstance(t1, str):
                def _sec(x: str) -> float:
                    parts = [float(p) for p in x.split(":")]
                    return parts[0] * 3600 + (parts[1] if len(parts) > 1 else 0) * 60 + parts[-1]
                per_step.append(_sec(t1) - _sec(t0))
    step_axis = [logs[i].get("current_steps") or logs[i].get("epoch") for i in range(len(per_step))]
    if step_axis and not isinstance(step_axis[-1], (int, float)):
        step_axis = list(range(1, len(per_step) + 1))

    # ---- 摘要 ----
    total_steps = state.get("global_step") or (steps[-1] if steps else 0)
    train_loss = results.get("train_loss")
    eval_loss = results.get("eval_loss")
    train_runtime = results.get("train_runtime")
    samples_per_sec = results.get("train_samples_per_second")
    last_loss = losses[-1] if losses else None
    last_eval = eval_losses[-1] if eval_losses else None

    print("===== 训练指标摘要 =====")
    print(f"输出目录    : {out_dir}")
    print(f"总步数      : {total_steps:,}")
    if train_loss is not None:
        print(f"最终 train_loss : {train_loss:.4f}")
    if last_eval is not None:
        print(f"最后 eval_loss  : {last_eval:.4f}")
    if eval_loss is not None:
        print(f"all_results eval_loss : {eval_loss:.4f}")
    if train_runtime:
        print(f"训练耗时    : {fmt_hms(train_runtime)}")
    if samples_per_sec:
        print(f"吞吐        : {samples_per_sec:.2f} 样本/秒")
    if last_loss is not None:
        print(f"末次打印 loss: {last_loss:.4f}")
    if len(steps) > 1:
        span = steps[-1] - steps[0]
        first_loss = next((x for x in losses if x is not None), None)
        if span and first_loss is not None and last_loss is not None:
            print(f"loss 从 {first_loss:.4f} → {last_loss:.4f}（{span:,} 步）")

    # ---- 绘图 ----
    # 默认写到仓库根 logs/ 下，用输出目录名命名（如 outputs/qwen3-4b-lora → logs/metrics_qwen3-4b-lora.png）
    if not args.out:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            f"metrics_{os.path.basename(os.path.normpath(out_dir))}.png",
        )
    else:
        out_path = args.out
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle("Training Metrics", fontsize=15, y=0.98)

    # 1. 训练 loss
    ax = axes[0, 0]
    ax.plot(steps, losses, lw=1.2, color="#1f77b4")
    ax.set_title("Train Loss")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    if last_loss is not None:
        ax.axhline(last_loss, color="#999", ls="--", lw=0.8)
        ax.annotate(f"{last_loss:.3f}", xy=(steps[-1], last_loss), xytext=(-70, 8),
                    textcoords="offset points", fontsize=9, color="#666")

    # 2. 验证 loss
    ax = axes[0, 1]
    if eval_steps:
        ax.plot(eval_steps, eval_losses, lw=1.2, marker="o", ms=3, color="#d62728")
        ax.set_title("Eval Loss")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if last_eval is not None:
            ax.axhline(last_eval, color="#999", ls="--", lw=0.8)
    else:
        ax.text(0.5, 0.5, "no eval records", ha="center", va="center", color="#999")
        ax.set_title("Eval Loss")

    # 3. 学习率
    ax = axes[1, 0]
    if lrs and len(lrs) == len(steps):
        ax.plot(steps, lrs, lw=1.2, color="#2ca02c")
    else:
        ax.text(0.5, 0.5, "no lr data", ha="center", va="center", color="#999")
    ax.set_title("Learning Rate")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # 4. 梯度范数
    ax = axes[1, 1]
    if grad_norms:
        ax.plot([s for s, _ in grad_norms], [g for _, g in grad_norms], lw=1.2, color="#9467bd")
        ax.set_title("Grad Norm")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no grad_norm", ha="center", va="center", color="#999")
        ax.set_title("Grad Norm")

    # 5. 每段耗时
    ax = axes[2, 0]
    if per_step:
        ax.plot(step_axis, per_step, lw=1.2, color="#ff7f0e")
        ax.set_title("Seconds per Log Interval")
        ax.set_xlabel("log interval #")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no timing data", ha="center", va="center", color="#999")
        ax.set_title("Seconds per Log Interval")

    # 6. 汇总文本
    ax = axes[2, 1]
    ax.axis("off")
    lines = [
        f"steps: {total_steps:,}",
        f"train loss: {train_loss:.4f}" if train_loss is not None else "train loss: -",
        f"eval loss: {eval_loss:.4f}" if eval_loss is not None else "eval loss: -",
        f"runtime: {fmt_hms(train_runtime)}" if train_runtime else "runtime: -",
        f"throughput: {samples_per_sec:.2f} samples/s" if samples_per_sec else "throughput: -",
        f"last logged loss: {last_loss:.4f}" if last_loss is not None else "last loss: -",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n图表已保存：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
