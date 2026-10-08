---
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

用《密教模拟器》(Cultist Simulator) 与《司辰之书》(Book of Hours) 的游戏内文本微调 [{base}](https://huggingface.co/{base}) 得到的 {params} 文本生成模型。目标是让它学会这两款游戏那种晦涩、克制、带书卷气又带点不祥暗示的写法，能写卡牌描述、事件叙事、结局文本，也能做中英互译。生成的文字可以直接拿去当同人创作、跑团素材或模组草稿。

## 模型详情

基座是 [{base}](https://huggingface.co/{base})，参数量 {params}，用 LoRA 微调（`r=16` / `alpha=32`，不是 QLoRA，权重为 bf16），训练框架是 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 0.9.4。支持中文和英文，许可跟随基座为 Apache-2.0。

仓库根目录放的是合并后的完整模型，LoRA 已经并入权重，`from_pretrained` 直接可加载，大小 {merged_size}；`lora/` 子目录放 LoRA 适配器（`r=16`），配合基座模型使用，大小 {lora_size}。按需要取其中一个就行。

## 训练细节

| 项目 | 值 |
| --- | --- |
| 基座模型 | [{base}](https://huggingface.co/{base}) |
| 微调方式 | LoRA（`r=16`，`alpha=32`，作用于 q/k/v/o/gate/up/down 全部投影层） |
| 训练样本 | {train_samples} 条（验证集 {eval_samples} 条，按实体 id 稳定划分） |
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
    "你是《密教模拟器》（Cultist Simulator）与《司辰之书》（Book of Hours）的文本生成模型。"
    "你精通这两款游戏晦涩、克制、带着书卷气与不祥暗示的文风，"
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

> eval_loss 也印证这个差距：4B 1.633 vs 0.6B 2.406。

## 数据来源与使用限制

训练语料全部抽取自《密教模拟器》与《司辰之书》两款游戏本体，版权归 Weather Factory 所有，中文文本来自游戏的官方简体中文资源。基座模型 {base} 采用 Apache-2.0 协议，本仓库的权重是它的微调衍生品。

请只把本模型用于个人学习、同人创作这类非商业用途。模型生成的内容有可能重现或近似游戏原文，公开发布生成结果时请遵守游戏原作者的授权条款并注明题材来源。另外它只学了文风和文本结构，并不理解也不会执行游戏机制，生成内容不保证与官方设定一致。

## 已知局限

训练数据只含叙事性文本字段，模型不知道卡牌的实际效果、配方条件与数值。因为训练集来自游戏本体，它有一定概率逐字输出或近似改写官方文本，直接用在作品里之前请自行确认授权。0.6B 这种小尺寸在需要严格保留格式的任务上会力不从心，比如中译英保留 `<b>` 标记就会丢，对格式遵循度要求高的场景建议用 4B。语言上只覆盖简体中文和英文，其他语言没训过。输出长度还受训练时 `cutoff_len=896` 影响，明显更长的连续文本不是它的强项。

## 复现

完整的数据抽取、训练、推理与发布流程见 GitHub 仓库：

```bash
git clone https://github.com/{github}.git
cd cultist-llm

# 抽取游戏文本 → 构建多任务数据集（需要本地已安装游戏本体）
.\scripts\prepare_data.ps1

# 训练 + 合并
.\scripts\train.ps1 -Size {size} -Merge
```

## 同系列模型

| 尺寸 | 仓库 |
| --- | --- |
| 0.6B | [`{ns}/cultist-simulator-qwen3-0.6b`](https://huggingface.co/{ns}/cultist-simulator-qwen3-0.6b) |
| 4B | [`{ns}/cultist-simulator-qwen3-4b`](https://huggingface.co/{ns}/cultist-simulator-qwen3-4b) |

## 相关链接

- 源码与训练脚本：<https://github.com/{github}>
- 基座模型：<https://huggingface.co/{base}>
