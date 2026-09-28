"""用微调后的模型生成《密教模拟器》文本。

支持三种权重形态：
  1. 直接加载合并后的完整模型      --model outputs/qwen3-4b-lora-merged
  2. 基座 + LoRA 适配器           --model models/Qwen3-4B --adapter outputs/qwen3-4b-lora
  3. 4bit 量化加载（省显存）       --model ... --load-4bit

用法：
    # 跑内置的一组示例提示词，看各类任务的效果
    python infer/generate.py --model models/Qwen3-4B --adapter outputs/qwen3-4b-lora

    # 交互式
    python infer/generate.py --model outputs/qwen3-4b-lora-merged --interactive

    # 用自定义提示词（jsonl，字段 instruction / input）
    python infer/generate.py --model ... --prompts my_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

SYSTEM_PROMPT = (
    "你是《密教模拟器》（Cultist Simulator）的文本生成模型。"
    "你精通该游戏晦涩、克制、带着书卷气与不祥暗示的文风，"
    "能够按给定的题材、类别和主题创作中文与英文游戏文本，也能在中英文之间准确互译。"
)

# 内置示例提示词，覆盖数据集里的各类任务
DEMO_PROMPTS = [
    {
        "instruction": "为《密教模拟器》中的一张卡牌撰写描述文本。",
        "input": "语言：中文\n卡牌名称：无光的陈列室\n类别：地点\n主题：冬、灯",
    },
    {
        "instruction": "以《密教模拟器》的文风，围绕给定主题写一段卡牌描述文本。",
        "input": "语言：中文\n主题：启",
    },
    {
        "instruction": "为《密教模拟器》撰写一次行动的开场叙述文本。",
        "input": "语言：中文\n行动类型：入梦\n事件名称：通往蜘蛛之门",
    },
    {
        "instruction": "为《密教模拟器》撰写一次行动结算后的叙述文本。",
        "input": "语言：中文\n行动类型：研究\n事件名称：翻译富奇诺语",
    },
    {
        "instruction": "撰写《密教模拟器》的一个结局正文。",
        "input": "语言：中文\n结局名称：共度余生：萝丝",
    },
    {
        "instruction": "下面是一段《密教模拟器》文本的开头，请以原文的风格与语气把它续写完整。",
        "input": "在梦中，我走过一条没有尽头的走廊。两侧的门都虚掩着，",
    },
    {
        "instruction": "把下面的《密教模拟器》中文文本翻译成英文。保留 <b>、<i> 等富文本标记与换行结构。",
        "input": "光透过<b>裂缝漏入</b>。我的头脑比<b>任何时候</b>都<b>更清晰</b>。",
    },
    {
        "instruction": "Write the description text for a card in Cultist Simulator.",
        "input": "Language: English\nCard name: The Unlit Gallery\nCategory: Location\nTheme: Winter, Lantern",
    },
]


def build_input_text(tokenizer, prompt: dict) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user = prompt["instruction"]
    if prompt.get("input"):
        user = f"{user}\n\n{prompt['input']}"
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load_model(args):
    print(f"[加载] 模型：{args.model}（device={args.device}）")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "device_map": args.device}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.float32 if args.device == "cpu" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel

        print(f"[加载] LoRA 适配器：{args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, text: str, args) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="生成《密教模拟器》风格文本")
    ap.add_argument("--model", required=True, help="基座模型或合并后的模型目录")
    ap.add_argument("--adapter", default="", help="LoRA 适配器目录（可选）")
    ap.add_argument("--prompts", default="", help="自定义提示词 jsonl（字段 instruction/input）")
    ap.add_argument("--interactive", action="store_true", help="交互式对话")
    ap.add_argument("--load-4bit", action="store_true", help="4bit 量化加载以节省显存")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu；cpu 可与 GPU 训练并行不抢算力")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条示例")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args)

    if args.interactive:
        print("\n交互模式：直接输入你的要求，留空行结束输入，Ctrl+C 退出。\n")
        while True:
            try:
                instruction = input("要求> ").strip()
                if not instruction:
                    continue
                extra = input("补充(可留空)> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            text = build_input_text(tokenizer, {"instruction": instruction, "input": extra})
            t0 = time.time()
            out = generate(model, tokenizer, text, args)
            print(f"\n{out}\n({time.time() - t0:.1f}s)\n" + "-" * 60)
        return

    if args.prompts:
        with open(args.prompts, encoding="utf-8") as fh:
            prompts = [json.loads(line) for line in fh if line.strip()]
    else:
        prompts = DEMO_PROMPTS
    if args.limit:
        prompts = prompts[: args.limit]

    for i, prompt in enumerate(prompts, 1):
        text = build_input_text(tokenizer, prompt)
        t0 = time.time()
        out = generate(model, tokenizer, text, args)
        print(f"\n{'=' * 70}\n[{i}/{len(prompts)}] {prompt['instruction']}")
        if prompt.get("input"):
            print("--- 输入 ---")
            print(prompt["input"])
        print("--- 生成 ---")
        print(out)
        print(f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()