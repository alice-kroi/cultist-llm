# cultist-llm —— 《密教模拟器》文本生成模型

## 1. 简述内容

用《密教模拟器》(Cultist Simulator) 的游戏内文本微调 Qwen3 系列模型，让它学会这个游戏那种晦涩、克制、带书卷气又带点不祥暗示的写法：能写卡牌描述、事件叙事、结局文本，也能做中英互译。已经训练并发布了 0.6B 和 4B 两个尺寸，下载地址见第 8 节。

语料由脚本从你本地安装的游戏里抽取，不依赖任何联网数据源，仓库里不含任何游戏原文，所以你需要自备游戏本体。代码以 MIT 发布，游戏文本版权归 [Weather Factory](https://weatherfactory.biz/) 所有。

## 2. 文件结构

```text
cultist-llm/
├── .gitignore              忽略权重、训练产物、数据集与日志
├── LICENSE                 MIT（只覆盖代码，不含游戏文本）
├── data/
│   ├── cs_json.py          容错 JSON 读取器（游戏的文件不是严格 JSON）
│   ├── extract.py          抽取游戏文本 → 中英对照语料
│   ├── build_dataset.py    语料 → 多任务指令数据集（alpaca 格式）
│   ├── corpus/             [生成物] text_units.jsonl / aspect_labels.json / stats.json
│   └── dataset/            [生成物] train.json / eval.json / dataset_info.json
├── configs/                5 个尺寸的 LLaMA-Factory 训练配置
├── templates/
│   └── model_card.md       Hugging Face 模型卡模板（上传时渲染成仓库首页）
├── scripts/
│   ├── prepare_data.ps1    一键：抽取 + 构建数据集
│   ├── download_models.py  从 ModelScope 下载 Qwen3 各尺寸
│   ├── train.ps1           一键：训练单个尺寸（可选合并 LoRA）
│   ├── train_all.ps1       一键：按顺序训练多个尺寸（16GB 显存只能串行）
│   └── upload_hf.py        上传到 Hugging Face（每尺寸一个仓库）
├── infer/generate.py       批量示例生成 / 交互式对话
├── models/                 [生成物] 基座权重（已 gitignore）
└── outputs/                [生成物] LoRA 权重（已 gitignore）
```

## 3. 数据集来源

全部语料来自本地已安装的《密教模拟器》。游戏把内容以 JSON 存在 `StreamingAssets/content/` 下，英文原文在 `core/`，简体中文翻译在 `loc_zh-hans/`，两者按实体 `id` 一一对应。

抽取结果（实测）：

| 指标 | 数值 |
| --- | --- |
| 文本单元 | 8,691 条（去重丢弃 2,438 条重复文本） |
| 中文字符 | 237,422 |
| 英文字符 | 831,834 |
| 中英成对 | 8,688 条（99.97%） |
| 理念(aspect) 索引 | 433 个，其中 307 个有中文名 |

覆盖的文本字段如下，其余字段像 `aspects`、`effects`、`slots`、`alt` 属于机制数据，不参与训练：

| 实体 | 文本字段 |
| --- | --- |
| elements | label, description |
| recipes | label, startdescription, description |
| endings | label, description, flavour |
| legacies | label, description, startdescription |
| verbs / decks | label, description |
| achievements | label, descriptionunlocked / descriptionlocked / unlockmessage |

文本里的 `<b>` / `<i>` / `<br>` 富文本标记和 `#PREVIOUSCHARACTERNAME#` 占位符按原样保留，它们就是游戏实际的显示格式。

每条语料中英文各成一条样本，最终 26,875 条训练样本 + 779 条验证样本，验证集按实体 id 稳定划分，同一实体的样本不会跨集：

| 任务 | 样本数 | 说明 |
| --- | --- | --- |
| `trans.zh2en` / `trans.en2zh` | 7233 / 7233 | 中英互译 |
| `gen.recipes.startdescription.*` | 1946 / 1964 | 行动开场叙事（给定行动类型 + 事件名） |
| `continue.zh` | 1953 | 给定开头按原文风格续写 |
| `gen.elements.description.*` | 1584 / 1596 | 卡牌描述（给定卡牌名 + 类别 + 主题） |
| `name.elements.zh` | 991 | 由主题设计新卡牌名称 |
| `gen.recipes.description.*` | 933 / 937 | 行动结算叙事 |
| `theme.elements.*` | 450 / 449 | 以准则（灯/铸/刃/冬/心/杯/蛾/启/秘史）为主题创作 |
| 其余（结局/成就/传承/牌堆/行动说明） | 442 | 长尾任务 |

语料不入库：`data/corpus/` 与 `data/dataset/` 已加入 `.gitignore`，请勿公开分发。

## 4. 基础环境

训练用的是 `llama-factory` conda 环境：torch 2.9.0+cu126 / transformers 4.57.1 / LLaMA-Factory 0.9.4，里面已经有 datasets、peft、modelscope。

只有 QLoRA(8B/14B) 需要 bitsandbytes，0.50.2 即可：

```powershell
conda run -n llama-factory pip install --no-deps "bitsandbytes>=0.45.0"
```

训练在一张 16GB 显存的 GPU 上完成，支持 bf16。配置里统一写 `flash_attn: sdpa`，不依赖 flash-attn。

## 5. 如何运行

准备数据、下基座、跑推理：

```powershell
git clone https://github.com/alice-kroi/cultist-llm.git
cd cultist-llm

# 数据源是你本地已安装的《密教模拟器》，用 -GameContent 指定游戏目录，大约 30 秒
.\scripts\prepare_data.ps1 -GameContent "X:\Games\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content"

# 下载基座模型
conda run -n llama-factory python scripts\download_models.py --sizes 4b
```

| 尺寸 | 权重占用（实测） | ModelScope id |
| --- | --- | --- |
| 0.6B | 1.41 GB | `Qwen/Qwen3-0.6B` |
| 1.7B | 3.80 GB | `Qwen/Qwen3-1.7B` |
| 4B | 7.51 GB | `Qwen/Qwen3-4B` |
| 8B | ~16 GB | `Qwen/Qwen3-8B` |
| 14B | ~28 GB | `Qwen/Qwen3-14B` |

权重要放到别的盘就加 `--models-dir X:\models\cultist-llm`，训练时用 `-ModelDir` 指同一个目录。

推理：

```powershell
# 跑内置的 8 条示例提示词（覆盖各类任务，含中英）
conda run -n llama-factory python infer/generate.py --model models\Qwen3-4B --adapter outputs\qwen3-4b-lora

# 交互式
conda run -n llama-factory python infer/generate.py --model outputs\qwen3-4b-lora-merged --interactive

# 8B 用 4bit 加载省显存
conda run -n llama-factory python infer/generate.py --model outputs\qwen3-8b-qlora-merged --load-4bit

# 自定义提示词（jsonl，字段 instruction / input）
conda run -n llama-factory python infer/generate.py --model ... --prompts my_prompts.jsonl
```

内置的示例提示词固定不变，适合拿同一组提示词横向对比不同尺寸。

## 6. 如何微调

```powershell
.\scripts\train.ps1 -Size 4b                                  # 按尺寸自动选配置
.\scripts\train.ps1 -Size 8b                                  # 8B 走 QLoRA
.\scripts\train.ps1 -Size 4b -Merge                           # 训练后合并成完整模型
.\scripts\train.ps1 -Size 4b -ModelDir X:\models\cultist-llm  # 权重在别的盘
.\scripts\train.ps1 -Size 4b -Extra "num_train_epochs=5 learning_rate=5e-5"
```

想一次排完多个尺寸就用 `train_all.ps1`（16GB 显存下必须串行，同时只能跑一个），`-Sizes` 要用逗号分隔：

```powershell
.\scripts\train_all.ps1 -Sizes 0.6b,1.7b,4b -Merge
.\scripts\train_all.ps1 -Sizes 8b,14b -Merge -ModelDir X:\cultist-llm-models
.\scripts\train_all.ps1 -Sizes 4b -Merge -SkipExisting   # 已有合并结果就跳过
```

| 配置 | 方式 | cutoff_len | batch×accum | epochs | 显存占用（未标「实测」者均为估算） |
| --- | --- | --- | --- | --- | --- |
| [qwen3_0.6b_lora.yaml](configs/qwen3_0.6b_lora.yaml) | LoRA bf16 | 896 | 4×4 | 4 | 15.9 GB VRAM / 1.0 s/步（实测） |
| [qwen3_1.7b_lora.yaml](configs/qwen3_1.7b_lora.yaml) | LoRA bf16 | 896 | 4×4 | 4 | ~10–13 GB |
| [qwen3_4b_lora.yaml](configs/qwen3_4b_lora.yaml) | LoRA bf16 | 896 | 1×16 | 3 | 14.3 GB（实测），余量仅 1.7 GB |
| [qwen3_8b_qlora.yaml](configs/qwen3_8b_qlora.yaml) | QLoRA 4bit | 896 | 1×16 | 3 | ~8–10 GB |
| [qwen3_14b_qlora.yaml](configs/qwen3_14b_qlora.yaml) | QLoRA 4bit | 896 | 1×16 | 3 | ~12–14 GB |

统一设置是 `lora_rank=16` / `alpha=32` / `lora_target=all`，等效 batch 16，`learning_rate=1e-4`，cosine 调度，`bf16` 加梯度检查点，每 epoch 约 1680 步。模板用 `qwen3_nothink`——本任务是风格化文本生成，不需要 Qwen3 的思考链。

`cutoff_len: 896` 是按真实数据定的：26,875 条训练样本的 token 长度最长 768、p99 310、均值 140，896 已覆盖 100%。4B 另开了 `pure_bf16: true`，让 LoRA 分支保持半精度（LLaMA-Factory 默认会降到 float32，显存翻倍）。

`-Merge` 默认在 CPU 上做，不吃显存；实测 0.6B 合并约 30 秒，产物是单个 bf16 的 `model.safetensors`，可以脱离 LoRA 适配器直接加载。

训练被 Ctrl+C 或关机打断后，加 `overwrite_output_dir=false` 重跑就行，LLaMA-Factory 会自动找 `output_dir` 下最新的 checkpoint 接着训：

```powershell
.\scripts\train.ps1 -Size 0.6b -Merge -Extra "overwrite_output_dir=false"
```

`-Extra` 用的是 LLaMA-Factory 原生的 OmegaConf 覆盖语法，写 `key=value`，不是 `--key value`。

## 7. 测试结果

| 尺寸 | 训练步数 | 训练时长 | eval_loss | 合并模型 | LoRA 适配器 |
| --- | --- | --- | --- | --- | --- |
| 0.6B | 6720 | 1 小时 22 分 | 2.406 | 1.13 GB | 38.5 MB |
| 4B | 5040 | 6 小时 | 1.949 | 7.51 GB | 126.1 MB |

两个尺寸的合并模型都用 CPU 加载跑过一遍生成验证，日志在 [`logs/verify_0.6b_merged.log`](logs/verify_0.6b_merged.log) 和 [`logs/verify_4b_merged.log`](logs/verify_4b_merged.log)。效果上 0.6B 格式遵循度一般，4B 的语感和格式遵循度都明显更好，所以推荐用 4B。

验证集是 779 条样本，按实体 id 与训练集隔离，同一实体的样本不会跨集。

## 8. 模型下载地址

两个仓库都已公开，根目录是合并后的完整模型（`from_pretrained` 直接加载），`lora/` 子目录是 LoRA 适配器（`PeftModel.from_pretrained(base, repo, subfolder="lora")`）：

- 0.6B：<https://huggingface.co/Cecilis/cultist-simulator-qwen3-0.6b>，1.18 GB，最轻量
- 4B：<https://huggingface.co/Cecilis/cultist-simulator-qwen3-4b>，7.65 GB，推荐

基座模型从 ModelScope 下载，见第 5 节。想自己重新发布一遍用 `scripts/upload_hf.py`。

## 9. 求星

如果这个项目对你有帮助，欢迎在 <https://github.com/alice-kroi/cultist-llm> 点个 Star，这对我很重要。有问题或想法也欢迎提 Issue。
