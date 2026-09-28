# cultist-llm —— 《密教模拟器》文本生成模型

用《密教模拟器》(Cultist Simulator) 的游戏内文本微调 Qwen3 系列模型，让它学会这个游戏那种晦涩、克制、带书卷气又带点不祥暗示的写法，能写卡牌描述、事件叙事、结局文本，也能做中英互译。

语料全部来自本地已安装的游戏，不依赖任何联网数据源。

> 你需要自备游戏本体。训练语料由脚本从你本地安装的《密教模拟器》里抽取，仓库里不含任何游戏原文，详见文末「版权与许可」。

## 已发布模型

两个仓库都已公开：

- 0.6B：<https://huggingface.co/Cecilis/cultist-simulator-qwen3-0.6b>，1.18 GB，最轻量，格式遵循度一般
- 4B：<https://huggingface.co/Cecilis/cultist-simulator-qwen3-4b>，7.65 GB，风格和格式遵循度都明显更好，推荐

两边都是根目录放合并后的完整模型（`from_pretrained` 直接加载），`lora/` 子目录放 LoRA 适配器。

## 目录结构

```
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
├── scripts/
│   ├── prepare_data.ps1    一键：抽取 + 构建数据集
│   ├── download_models.py  从 ModelScope 下载 Qwen3 各尺寸
│   ├── train.ps1           一键：训练单个尺寸（可选合并 LoRA）
│   ├── train_all.ps1       一键：按顺序训练多个尺寸（16GB 显存只能串行）
│   └── upload_hf.py        上传到 Hugging Face（每尺寸一个仓库，含自动生成的模型卡）
├── infer/generate.py       批量示例生成 / 交互式对话
├── models/                 [生成物] 基座权重（已 gitignore）
└── outputs/                [生成物] LoRA 权重（已 gitignore）
```

## 快速开始

```powershell
git clone https://github.com/alice-kroi/cultist-llm.git
cd cultist-llm

# 数据源是你本地已安装的《密教模拟器》，用 -GameContent 指定游戏目录，见「2. 数据准备」

# 数据准备，大约 30 秒
.\scripts\prepare_data.ps1 -GameContent "<Steam 库>\steamapps\common\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content"

# 下载基座模型（4B 约 8GB；14B 需要约 28GB 磁盘，见「3. 模型下载」）
conda run -n llama-factory python scripts\download_models.py --sizes 4b

# 训练
.\scripts\train.ps1 -Size 4b

# 看效果
conda run -n llama-factory python infer/generate.py --model models\Qwen3-4B --adapter outputs\qwen3-4b-lora
```

## 1. 环境

训练用的是 `llama-factory` conda 环境：torch 2.9.0+cu126 / transformers 4.57.1 / LLaMA-Factory 0.9.4，里面已经有 datasets、peft、modelscope。

只有 QLoRA(8B/14B) 需要 bitsandbytes，0.50.2 即可：

```powershell
conda run -n llama-factory pip install --no-deps "bitsandbytes>=0.45.0"
```

加 `--no-deps` 是因为 torch/numpy 环境本来就齐了，不想让 pip 顺带动它们。装完 `import bitsandbytes` 和 `torch.cuda.is_available()` 都是 True，也用 0.6B 冒烟验证过 4bit 量化训练能跑通，日志里出现了 `Quantizing model to 4 bit with bitsandbytes.`。

训练在一张 16GB 显存的 GPU 上完成，支持 bf16。配置里统一写 `flash_attn: sdpa`，不依赖 flash-attn。

## 2. 数据准备

### 数据来源

游戏把全部内容以 JSON 存在 `StreamingAssets/content/` 下，英文原文在 `core/`，简体中文翻译在 `loc_zh-hans/`，两者按实体 `id` 一一对应。

用 `-GameContent` 指定游戏目录（Steam 默认安装位置形如 `<Steam 库>\steamapps\common\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content`）：

```powershell
.\scripts\prepare_data.ps1 -GameContent "X:\Games\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content"
```

抽出的是 `core/` 143 个文件 3827 个实体、`loc_zh-hans/` 163 个文件 3651 个实体，中文覆盖面 99.7%。

### 三个要处理的坑

**坑 1：游戏 JSON 不是合法 JSON。** 全库 1170 个文件里有 25 个标准 `json` 模块读不了，原因是三类问题：无引号键名（`{id: "stagriddles"}`）、尾随逗号、字符串里带裸换行或制表符。另外还有键名大小写混用（`actionId` 和 `actionid` 并存、`descriptionunlocked`）和一个键名里嵌了换行的脏数据。`data/cs_json.py` 用一次性字符扫描做保守修复，只动这三类位置、不改字符串内容；`extract.py` 里读字段全部走大小写和空白归一化。这样 1170 个文件现在都能解析。

**坑 2：不是所有中文条目都真的翻译了。** 有 84 条是内部标记实体，内容像 `"False trail marker: DO NOT TRANSLATE"`，构建数据集时按「中文侧必须含汉字」过滤掉了。

**坑 3：中英两侧的富文本标记并不对应。** 中文侧习惯用 `<b>` 标重点，英文原文大多没有对应标记。如果翻译任务一律要求「保留 `<b>`、`<i>` 标记」，就会造出指令要求保留、参考译文里却根本没有标记的自相矛盾样本——实测 687 条带标记的中译英样本里有 541 条答案没有标记，等于在教模型把标记丢掉。`build_dataset.py` 的办法是：两侧都带标记才要求保留，否则把两侧标记一并剥掉，保证每条样本的指令和答案自洽。修复后 146 条带标记样本 100% 一致，矛盾样本 0 条。

### 抽取结果（实测）

| 指标 | 数值 |
| --- | --- |
| 文本单元 | 8,691 条（去重丢弃 2,438 条重复文本） |
| 中文字符 | 237,422 |
| 英文字符 | 831,834 |
| 中英成对 | 8,688 条（99.97%） |
| 理念(aspect) 索引 | 433 个，其中 307 个有中文名 |

覆盖的字段如下，其余字段像 `aspects`、`effects`、`slots`、`alt` 都属于机制数据，不参与训练：

| 实体 | 文本字段 |
| --- | --- |
| elements | label, description |
| recipes | label, startdescription, description |
| endings | label, description, flavour |
| legacies | label, description, startdescription |
| verbs / decks | label, description |
| achievements | label, descriptionunlocked / descriptionlocked / unlockmessage |

`recipes` 的 `alt` 字段是 `[{"id": ..., "chance": ...}]` 形式的跳转引用，不含文本，已忽略。文本里的 `<b>` / `<i>` / `<br>` 富文本标记和 `#PREVIOUSCHARACTERNAME#` 占位符都按原样保留，它们就是游戏实际的显示格式。

### 训练任务设计

每条语料可以派生多个样本，中英文各成一条，最后是 26,875 条训练样本 + 779 条验证样本。验证集按实体 id 稳定划分，同一个实体的样本不会跨集；输入加输出超过 1600 字符的 56 条样本已丢弃，免得被 `cutoff_len` 截断后污染训练目标。

| 任务 | 样本数 | 说明 |
| --- | --- | --- |
| `trans.zh2en` / `trans.en2zh` | 7233 / 7233 | 中英互译；两侧都带富文本标记时才要求保留，否则剥掉标记 |
| `gen.recipes.startdescription.*` | 1946 / 1964 | 行动开场叙事（给定行动类型 + 事件名） |
| `continue.zh` | 1953 | 给定开头按原文风格续写 |
| `gen.elements.description.*` | 1584 / 1596 | 卡牌描述（给定卡牌名 + 类别 + 主题） |
| `name.elements.zh` | 991 | 由主题设计新卡牌名称 |
| `gen.recipes.description.*` | 933 / 937 | 行动结算叙事 |
| `theme.elements.*` | 450 / 449 | 以准则（灯/铸/刃/冬/心/杯/蛾/启/秘史）为主题创作 |
| 其余（结局/成就/传承/牌堆/行动说明） | 442 | 长尾任务 |

表里是去重前各任务派生的样本数，合计 27,711；去重丢掉 57 条重复样本后落到 26,875 + 779。

提示词里的「类别」和「主题」来自实体的 aspect，通过 `aspect_labels.json` 映射成中文名（`lantern` → 灯，`lore` → 密传），所以提示词是可读的自然语言，不是内部 id。

## 3. 模型下载

```powershell
conda run -n llama-factory python scripts\download_models.py --sizes 0.6b 1.7b 4b 8b
conda run -n llama-factory python scripts\download_models.py --all --models-dir X:\models\cultist-llm
```

| 尺寸 | 权重占用（实测） | ModelScope id |
| --- | --- | --- |
| 0.6B | 1.41 GB | `Qwen/Qwen3-0.6B` |
| 1.7B | 3.80 GB | `Qwen/Qwen3-1.7B` |
| 4B | 7.51 GB | `Qwen/Qwen3-4B` |
| 8B | ~16 GB | `Qwen/Qwen3-8B` |
| 14B | ~28 GB | `Qwen/Qwen3-14B` |

跑完 0.6B 加 4B 之后，`models/` 占 12.72 GB，`outputs/` 清理 checkpoint 后占 9.34 GB（0.6B LoRA 0.18 + 0.6B merged 1.13 + 4B LoRA 0.52 + 4B merged 7.51）。中间产物 `outputs\qwen3-*-lora\checkpoint-*` 单个只有 0.13 GB（0.6B）/ 0.38 GB（4B），但架不住多：0.6B 有 34 个、4B 有 26 个，合计 14.3 GB。只保留各自最后一个（`checkpoint-6720` / `checkpoint-5040` 兜底），可以一次性回收 13.84 GB。LoRA 适配器本身在 `outputs\qwen3-*-lora\` 根目录（`adapter_model.safetensors`，0.6B 38.5 MB / 4B 126.1 MB，`r=16`），删 `checkpoint-*` 子目录不会碰到它，合并模型也单独验证过能加载。

所以想再下 8B（~16 GB）已经勉强，14B（~28 GB）肯定放不下。权重可以放到别的盘，用 `--models-dir <目录>` 指定、训练时用 `-ModelDir <目录>` 覆盖配置里的路径，脚本会预检空间并给出提示。

## 4. 训练

```powershell
.\scripts\train.ps1 -Size 4b                                  # 按尺寸自动选配置
.\scripts\train.ps1 -Size 8b                                  # 8B 走 QLoRA
.\scripts\train.ps1 -Size 4b -Merge                           # 训练后合并成完整模型
.\scripts\train.ps1 -Size 4b -ModelDir X:\models\cultist-llm  # 权重在别的盘
.\scripts\train.ps1 -Size 4b -Extra "num_train_epochs=5 learning_rate=5e-5"
```

想一次排完多个尺寸就用 `train_all.ps1`（16GB 显存下必须串行，同时只能跑一个）：

```powershell
.\scripts\train_all.ps1 -Sizes 0.6b,1.7b,4b -Merge
.\scripts\train_all.ps1 -Sizes 8b,14b -Merge -ModelDir X:\cultist-llm-models
.\scripts\train_all.ps1 -Sizes 4b -Merge -SkipExisting   # 已有合并结果就跳过
```

`-Sizes` 要用逗号分隔：`powershell -File` 传参时空格分隔的数组只会绑到第一个值，脚本里已经对逗号串做了兜底拆分。每个尺寸的日志写到 `logs\train_<size>.log`，单个尺寸失败不会中断后面的尺寸，最后统一汇报失败列表。

`-Merge` 默认在 CPU 上做：LLaMA-Factory 的 `export_device` 默认值就是 `cpu`，所以导出不吃显存、不抢 GPU 算力，训练还在跑的时候就能先把某个中间 checkpoint 合并出来验证。实测 0.6B 合并约 30 秒，产物是单个 1.14 GB 的 `model.safetensors`（bf16），可以脱离 LoRA 适配器直接加载：

```powershell
conda run -n llama-factory infer\generate.py --model outputs\qwen3-0.6b-lora-merged --device cpu
```

`-Extra` 用的是 LLaMA-Factory 原生的 OmegaConf 覆盖语法，写 `key=value`，不是 `--key value`。脚本本身不改配置文件，只把覆盖参数追加到命令行，所以不会引入编码问题。

**中断后续训**：训练被 Ctrl+C 或关机打断后，直接加 `overwrite_output_dir=false` 重跑就行，LLaMA-Factory 会自动找 `output_dir` 下最新的 checkpoint 接着训：

```powershell
.\scripts\train.ps1 -Size 0.6b -Merge -Extra "overwrite_output_dir=false"
```

也可以指定具体 checkpoint：`-Extra "resume_from_checkpoint=<输出目录>\checkpoint-800"`。配置里 `overwrite_output_dir: true` 只会清掉 `trainer_log.jsonl`，不会删 checkpoint，所以中断不会丢进度。

| 配置 | 方式 | cutoff_len | batch×accum | epochs | 显存占用（未标「实测」者均为估算） |
| --- | --- | --- | --- | --- | --- |
| [qwen3_0.6b_lora.yaml](configs/qwen3_0.6b_lora.yaml) | LoRA bf16 | 896 | 4×4 | 4 | 15.9 GB VRAM / 1.0 s/步（实测） |
| [qwen3_1.7b_lora.yaml](configs/qwen3_1.7b_lora.yaml) | LoRA bf16 | 896 | 4×4 | 4 | ~10–13 GB |
| [qwen3_4b_lora.yaml](configs/qwen3_4b_lora.yaml) | LoRA bf16 | 896 | 1×16 | 3 | 14.3 GB（实测），余量仅 1.7 GB |
| [qwen3_8b_qlora.yaml](configs/qwen3_8b_qlora.yaml) | QLoRA 4bit | 896 | 1×16 | 3 | ~8–10 GB |
| [qwen3_14b_qlora.yaml](configs/qwen3_14b_qlora.yaml) | QLoRA 4bit | 896 | 1×16 | 3 | ~12–14 GB |

显存不主要看参数量：LM head 对 15.1 万词表算 logits 才是大头，正比于 `per_device_train_batch_size × cutoff_len × vocab`，所以 4B 用小 batch 反而可能比 0.6B 用大 batch 更省。

统一设置是 `lora_rank=16` / `alpha=32` / `lora_target=all`，等效 batch 16，`learning_rate=1e-4`，cosine 调度，`bf16` 加梯度检查点，每 epoch 约 1680 步。

4B 另外开了 `pure_bf16: true`。LLaMA-Factory 默认会把 LoRA 可训练参数升到 float32（日志里是 `Upcasting trainable params to float32.`），LoRA 分支一旦是 fp32，加法就会把整条激活链和 LM head logits 一起提升成 fp32，显存直接翻倍。置为 `pure_bf16` 后 LoRA 保持半精度（日志变成 `Pure bf16 / BAdam detected, remaining trainable params in half precision.`）。`hparams/parser.py` 里 pure_bf16 的前置条件是「设备支持 bf16 且不用 DeepSpeed ZeRO-3」，满足即可开。0.6B 保持默认，跑通了且显存够用。

`cutoff_len: 896` 是按真实数据定的：实测 26,875 条训练样本的 token 长度最长 768、p99 310、均值 140，896 已覆盖 100%，再大纯属浪费显存和算力。

模板用 `qwen3_nothink`——本任务是风格化文本生成，不需要 Qwen3 的思考链。

显存不够时的调整顺序：先降 `per_device_train_batch_size` 并同步调大 `gradient_accumulation_steps`（保持等效 batch 不变），再降 `cutoff_len`，最后把 `lora_rank` 降到 8。判断健康用步速加整卡功耗，别用 `nvidia-smi` 的显存数字或 `Shared Usage`——显存超订时 WDDM 会把溢出到系统内存的部分也算进 used，数字顶到上限反而看不出差别。另外跑之前关掉吃 GPU 的桌面程序（Wallpaper Engine、模拟器等），它们不占显存但会抢算力，实测步速能差出约 8 倍。

## 5. 推理与效果验证

```powershell
# 跑内置的 8 条示例提示词（覆盖各类任务，含中英）
conda run -n llama-factory python infer/generate.py --model models\Qwen3-4B --adapter outputs\qwen3-4b-lora

# 交互式
conda run -n llama-factory python infer/generate.py --model outputs\qwen3-4b-lora-merged --interactive

# 8B 用 4bit 加载省显存
conda run -n llama-factory python infer/generate.py --model outputs\qwen3-8b-qlora-merged --load-4bit

# 用 CPU 跑（fp32）：不占显存、不抢 GPU 算力，可在训练进行中并行验证中间 checkpoint
conda run -n llama-factory python infer/generate.py --model models\Qwen3-0.6B --adapter outputs\qwen3-0.6b-lora\checkpoint-1200 --device cpu

# 自定义提示词（jsonl，字段 instruction / input）
conda run -n llama-factory python infer/generate.py --model ... --prompts my_prompts.jsonl
```

对比不同尺寸时，用 `-Merge` 分别合并后跑同一组提示词就行；`infer/generate.py` 内置的示例提示词固定不变，适合做横向对比。

## 6. 上传到 Hugging Face

已发布的公开仓库：

- <https://huggingface.co/Cecilis/cultist-simulator-qwen3-0.6b>
- <https://huggingface.co/Cecilis/cultist-simulator-qwen3-4b>

`scripts/upload_hf.py` 每个尺寸建一个仓库，仓库里是这样放的：

```text
<repo>/
    <合并后的完整模型>      # 根目录：from_pretrained 直接可加载
    lora/                   # LoRA 适配器（只传适配器与分词器，不含 checkpoint）
    README.md               # 模型卡，脚本自动生成
```

根目录放合并模型、适配器放 `lora/` 子目录，是为了让两种用法都不歧义：`from_pretrained(repo)` 拿完整模型，`PeftModel.from_pretrained(base, repo, subfolder="lora")` 拿适配器。

模型卡由脚本里的 `CARD` 模板生成，带完整 HF 元数据（`library_name`、`base_model`、`base_model_relation: finetune`、`license`、`language`、`pipeline_tag`、`tags`），正文含模型详情、仓库内容、快速使用（transformers / peft / Ollama）、训练细节、效果示例、已知局限和复现步骤，并自动回链到 GitHub 仓库与同系列其他尺寸。GitHub 地址写在脚本顶部的 `GITHUB_REPO` 常量里，换仓库只改这一处。

```powershell
# 先干跑：只列清单、统计体积，不发任何网络请求
conda run -n llama-factory python scripts\upload_hf.py --sizes 0.6b,4b --dry-run

# 实际上传（默认公开仓库，仓库名 <账号>/cultist-simulator-qwen3-<尺寸>）
conda run -n llama-factory python scripts\upload_hf.py --sizes 0.6b,4b --token hf_xxx

# 只传 LoRA / 建私有仓库 / 自定义仓库名
conda run -n llama-factory python scripts\upload_hf.py --sizes 4b --token hf_xxx --skip-merged
conda run -n llama-factory python scripts\upload_hf.py --sizes 0.6b,4b --token hf_xxx --private
conda run -n llama-factory python scripts\upload_hf.py --sizes 4b --token hf_xxx --repo-id me/my-model

# 只更新模型卡：改了卡片文案不必重传几 GB 权重
conda run -n llama-factory python scripts\upload_hf.py --sizes 0.6b,4b --token hf_xxx --card-only
```

实测体积：0.6B 合并 1.13 GB 加 LoRA 53.7 MB，4B 合并 7.51 GB 加 LoRA 141.2 MB，合计约 8.8 GB。

### 网络不通时的代理设置

如果直连 `huggingface.co` 不通（TLS 被重置、连接超时），就得走代理。注意 Python 的 `requests` 不读 Windows 的 WinINET 系统代理设置，只认 `HTTP_PROXY` / `HTTPS_PROXY`，所以要显式导出：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:<端口>"
$env:HTTP_PROXY  = "http://127.0.0.1:<端口>"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"   # 走经典分片上传
conda run --no-capture-output -n llama-factory python scripts\upload_hf.py `
    --sizes 0.6b --token hf_xxx --endpoint https://huggingface.co
```

实测上传约 6.5 MB/s，0.6B（1.19 GB）约 5 分钟，4B（7.65 GB）约 20 分钟。

## 版权与许可

| 对象 | 许可 / 权利归属 |
| --- | --- |
| 本项目代码（`data/`、`scripts/`、`configs/`、`infer/`） | [MIT](LICENSE) |
| 《密教模拟器》游戏文本 | 版权归 [Weather Factory](https://weatherfactory.biz/) 所有 |
| 微调后的模型权重 | 基座 Qwen3 为 Apache-2.0，衍生权重同样按 Apache-2.0 发布 |

三点需要留意：

1. 游戏文本不入库。语料由脚本从你本地已购买安装的游戏副本中抽取，`data/corpus/`、`data/dataset/` 已加入 `.gitignore`，请勿公开分发。
2. 模型输出可能重现原文。把生成内容公开发布时，请遵守游戏原作者的授权条款并注明题材来源；本项目面向个人学习与同人创作等非商业用途。
3. 复现数据准备需要自备游戏本体；没有游戏也可以直接使用已发布的模型权重。

## 致谢

基座模型 [Qwen3](https://github.com/QwenLM/Qwen3)（Apache-2.0）、训练框架 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，以及游戏文本与世界观来源《密教模拟器》 by Weather Factory。
