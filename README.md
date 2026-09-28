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

# 下载基座模型（4B 约 8GB；14B 需要约 28GB 磁盘，见「磁盘」）
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

这里面踩过的坑不少，按踩到的顺序记一下。

> **`train_all.ps1` 必须放宽 `$ErrorActionPreference`。** PowerShell 5.1 下 `& powershell @TrainArgs *> $LogFile` 会把子进程写到 stderr 的内容（比如 python 的 `UserWarning`，jieba 加载时就会打一条 `pkg_resources` 弃用警告）转成 `ErrorRecord`，而脚本开头设了 `$ErrorActionPreference = "Stop"`，于是这些普通警告被当成致命错误，把 `train_all.ps1` 自己干掉，子进程则变成孤儿继续在后台跑。症状挺有迷惑性：`train_all.ps1` 的输出停在 `================ 0.6b 开始 ================` 不动，而训练其实在正常推进。现在脚本在调用子进程期间临时把偏好放宽为 `Continue`，只按退出码判断成败（`train.ps1` 里 QLoRA 依赖探测的 `2>$null` 也做了同样处理）。

> **Windows 上不要开 `dataloader_num_workers`。** 配合 `gradient_checkpointing` 时，HF 的 `PreTrainedModel.enable_input_require_grads()` 注册的是本地闭包，而 Windows 的 DataLoader worker 用 spawn，需要 pickle 它，直接崩：`AttributeError: Can't pickle local object 'PreTrainedModel.enable_input_require_grads.<locals>.make_inputs_require_grads'`。这个崩的表现也很隐蔽：python 进程退出码 1、日志停在 `trainable params: ...` 之后，连 traceback 都未必留下。`preprocessing_num_workers`（数据预处理阶段）不受影响，可以保留。

> **`conda run` 会吞掉训练日志。** 不加 `--no-capture-output` 时，`conda run` 会把子进程的 stdout/stderr 全部缓存到进程结束才吐出来，于是 `logs\train_<size>.log` 在训练的整个过程中一直是空的（实测只写了脚本自己的几行 header，666 字节），出了错也看不到 traceback，只能靠 `trainer_log.jsonl` 猜。`train.ps1` 现在统一用 `conda run --no-capture-output`（等价 `--live-stream`），日志实时落盘。

`-Merge` 默认在 CPU 上做：LLaMA-Factory 的 `export_device` 默认值就是 `cpu`，所以导出不吃显存、不抢 GPU 算力，训练还在跑的时候就能先把某个中间 checkpoint 合并出来验证。实测 0.6B 合并约 30 秒，产物是单个 1.14 GB 的 `model.safetensors`（bf16），可以脱离 LoRA 适配器直接加载：

```powershell
conda run -n llama-factory infer\generate.py --model outputs\qwen3-0.6b-lora-merged --device cpu
```

`-Extra` 用的是 LLaMA-Factory 原生的 OmegaConf 覆盖语法，写 `key=value`，不是 `--key value`。脚本本身不改配置文件，只把覆盖参数追加到命令行，所以不会引入编码问题。

> 坑：OmegaConf 会把 `no` / `yes` / `true` / `false` 这类裸值解析成布尔。想覆盖成字符串枚举（比如关掉评估）不能写 `eval_strategy=no`，会报 `ValueError: False is not a valid IntervalStrategy`，改用 `do_eval=false`。

**中断后续训**：训练被 Ctrl+C 或关机打断后，直接加 `overwrite_output_dir=false` 重跑就行，LLaMA-Factory 会自动找 `output_dir` 下最新的 checkpoint 接着训：

```powershell
.\scripts\train.ps1 -Size 0.6b -Merge -Extra "overwrite_output_dir=false"
```

也可以指定具体 checkpoint：`-Extra "resume_from_checkpoint=<输出目录>\checkpoint-800"`。配置里 `overwrite_output_dir: true` 只会清掉 `trainer_log.jsonl`，不会删 checkpoint，所以中断不会丢进度。

> 坑（**改了 batch 再续训，batch 不生效**）：HF 的 `Trainer` 在续训时会把 `self._train_batch_size` 覆盖成 checkpoint 里 `trainer_state.json` 记的 `train_batch_size`（transformers/trainer.py 里那段注释写的是 "In case of repeating the find_executable_batch_size"）。也就是说 `per_device_train_batch_size` 不是每轮都从 yaml 读，而是被 checkpoint 的记录值顶掉。实测踩到的后果：把 0.6B 从 `8×2` 改成 `4×4` 后加 `overwrite_output_dir=false` 重启，日志里 "Instantaneous batch size per device = 4" 看着正常，但 dataloader 仍是按 batch 8 建的，于是 `Total optimization steps` 从 6720 变成 3360、实际等效 batch 是 32 而不是 16，显存又回到溢出区（Shared 12 GB、11 s/步、功耗 44 W）。修正办法：续训前把最新 checkpoint 的 `trainer_state.json` 里 `train_batch_size` 改成和 yaml 里 `per_device_train_batch_size` 一致，或者干脆清掉输出目录从头训。判断有没有中招，看启动日志这三行是否与预期一致：
>
> ```text
> Instantaneous batch size per device = 4
> Total train batch size (w. parallel, distributed & accumulation) = 16   # 4 × 4
> Total optimization steps = 6,720                                        # 1680 步/epoch × 4
> ```

| 配置 | 方式 | cutoff_len | batch×accum | epochs | 显存占用（未标「实测」者均为估算） |
| --- | --- | --- | --- | --- | --- |
| [qwen3_0.6b_lora.yaml](configs/qwen3_0.6b_lora.yaml) | LoRA bf16 | 896 | 4×4 | 4 | 15.9 GB VRAM / 1.0 s/步（实测） |
| [qwen3_1.7b_lora.yaml](configs/qwen3_1.7b_lora.yaml) | LoRA bf16 | 896 | 4×4 | 4 | ~10–13 GB |
| [qwen3_4b_lora.yaml](configs/qwen3_4b_lora.yaml) | LoRA bf16 | 896 | 1×16 | 3 | 14.3 GB（实测），余量仅 1.7 GB |
| [qwen3_8b_qlora.yaml](configs/qwen3_8b_qlora.yaml) | QLoRA 4bit | 896 | 1×16 | 3 | ~8–10 GB |
| [qwen3_14b_qlora.yaml](configs/qwen3_14b_qlora.yaml) | QLoRA 4bit | 896 | 1×16 | 3 | ~12–14 GB |

关于显存有几件事得说明白，不然很容易误判。

`nvidia-smi` 的数字只能当参考，不能当健康判据：0.6B 在 `4×4`（健康、1.0 s/步）和 `8×2`（溢出、9 s/步）这两种情况下，`nvidia-smi` 都报 15.6~15.9 / 16 GB，因为显存超订后 WDDM 会把溢出到系统内存的那部分也算进 used，数字直接顶到上限。真正的判据是步速加整卡功耗。

显存也不主要看参数量。LM head 对 15.1 万词表算 logits 才是大头，它正比于 `per_device_train_batch_size × cutoff_len × vocab`，所以 4B 用小 batch 反而可能比 0.6B 用大 batch 更省。降显存第一优先级是降 per-device batch，不是降模型尺寸：4B 若 OOM，改成 `batch=1 + accum=16`，等效 batch 不变、步数也不变。

具体到 0.6B：用 `8×896` 时 WDDM 会把 GPU 内存溢出到系统内存。出问题时同时出现三个特征——步速从约 1.2 s/步劣化到 9 s/步、整卡功耗从 120 W+ 掉到 42~80 W，而 `nvidia-smi` 的 GPU 占用率仍显示 100%，所以只看占用率会误判成"GPU 很忙"，同一时刻 `Get-Counter '\GPU Engine(*)'` 也会显示 python 占 99%，同样会误判。本质是显存超订后每个 kernel 都要走 PCIe 取溢出页。改成 `4×4` 重启后步速回到 1.0 s/步、功耗回到 115 W。但如果带着旧 checkpoint 续训，步速又会掉回 11 s/步——那不是换页重启没生效，而是上面那个 `train_batch_size` 的坑把 batch 顶回了 8（总步数从 6720 变 3360 是同一个坑的另一个信号）。把 `trainer_state.json` 的 `train_batch_size` 修回 4 再续训，步速恢复 1.0 s/步、功耗 115 W。

`Shared Usage` 也不能单独拿来当健康判据（这里原先写的「健康 ≈1 GB、溢出 15.9 GB」已被本轮实测推翻，更正一下）：这个计数器随进程存活时间爬升，和"是否正在换页"并不一一对应。实测同一个健康进程（1.0 s/步、96~98% 占用、约 100~110 W 功耗）`Shared Usage` 稳定在 10.96 GB 一动不动，而同一进程刚重启时读到的却是 1.0 GB。也就是说「Shared ≈1 GB 才健康」这种绝对阈值是错的，它只有配合步速和功耗一起看才有意义：Shared 顶到上限、且步速掉到 10 s/步量级、且功耗掉到 40~80 W，才是真的在换页。采集方式（只看占用 >0.1 GB 的进程）：

```powershell
(Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples |
  Where-Object { $_.CookedValue -gt 100000000 } |
  Select-Object InstanceName, @{n='GB';e={[math]::Round($_.CookedValue/1GB,2)}}
```

判断训练是否健康，看步速和功耗最直接：GFX 满载约 100~120 W+，若只有 40~80 W 就说明 GPU 在等待（换页或算力被抢）；此时 `trainer_log.jsonl` 里的 `elapsed_time` / `remaining_time` 也会失真，实测 step 2640/6720 时它报"剩余 19 分钟"，按 1.0 s/步实际还有约 68 分钟。

统一设置是 `lora_rank=16` / `alpha=32` / `lora_target=all`，等效 batch 16，`learning_rate=1e-4`，cosine 调度，`bf16` 加梯度检查点，每 epoch 约 1680 步。

4B 另外开了 `pure_bf16: true`。LLaMA-Factory 默认会把 LoRA 可训练参数升到 float32（日志里是 `Upcasting trainable params to float32.`），LoRA 分支一旦是 fp32，加法就会把整条激活链和 LM head logits 一起提升成 fp32，显存直接翻倍。置为 `pure_bf16` 后 LoRA 保持半精度（日志变成 `Pure bf16 / BAdam detected, remaining trainable params in half precision.`）。`hparams/parser.py` 里 pure_bf16 的前置条件是「设备支持 bf16 且不用 DeepSpeed ZeRO-3」，满足即可开。0.6B 保持默认，跑通了且显存够用。

`cutoff_len: 896` 是按真实数据定的：实测 26,875 条训练样本的 token 长度最长 768、p99 310、均值 140，896 已覆盖 100%，再大纯属浪费显存和算力。

模板用 `qwen3_nothink`——本任务是风格化文本生成，不需要 Qwen3 的思考链。

显存不够时的调整顺序：先降 `per_device_train_batch_size` 并同步调大 `gradient_accumulation_steps`（保持等效 batch 不变），再降 `cutoff_len`，最后把 `lora_rank` 降到 8。

最后还有一条容易忽略的：跑之前请关掉吃 GPU 的桌面程序（Wallpaper Engine、MuMu 模拟器、游戏启动器等），它们不占多少显存但会抢走 GPU 算力。现象是 0.6B 步速在 1.1 s/步与 8~10 s/步之间剧烈波动（SM 频率 2820 MHz 正常，但整卡功耗只有 60 W / 165 W，而不是满载的 120 W+），训练日志里的 `remaining_time` 会因此严重失真。对照实测：同一份配置，桌面上挂着 Wallpaper Engine / MuMu 时是 8~10 s/步，关掉后立刻回到 1.1 s/步，差了大约 8 倍。空闲时显存占用只有约 0.5 GB，所以这不是显存问题，是算力被抢。

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
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"   # 见坑 4
conda run --no-capture-output -n llama-factory python scripts\upload_hf.py `
    --sizes 0.6b --token hf_xxx --endpoint https://huggingface.co
```

实测上传约 6.5 MB/s，0.6B（1.19 GB）约 5 分钟，4B（7.65 GB）约 20 分钟。

上传时踩的坑有五个，都是实测出来的。

> 坑 1：`HF_TOKEN` 环境变量的优先级最高，高过显式传入的 `token=`。实测（huggingface_hub 0.35.3）`HfApi(token=新令牌).whoami()` 依然报 `Invalid user token`，因为环境变量里的旧令牌把它顶掉了。所以脚本在收到 `--token` 后会先 `os.environ.pop("HF_TOKEN")`，否则 `--token` 会静默失效。另外 `[Environment]::SetEnvironmentVariable(...,'User')` 只写注册表，当前已开的进程仍是旧值，`conda run` 会把这个旧值传给子进程。

> 坑 2：fine-grained 令牌的权限要单独授，默认是全空的，令牌有效不代表能上传。必须确认 `auth.accessToken.fineGrained.scoped[0].permissions` 里含 `repo.write`。注意权限词表里没有 `repo.create` 这一项，建仓库靠的就是 `repo.write`，拿 `repo.create` 去比对会永远显示"不可用"。这个权限检查失败的表现也很迷惑：裸调 `https://huggingface.co/api/whoami-v2` 返回 200，而 `huggingface_hub` 的 `HfApi().whoami()` 却报 `Invalid user token`。解决办法是在 <https://huggingface.co/settings/tokens> 新建令牌，要么直接用经典的 Write 类型（最省事），要么建 fine-grained 时把 `Repositories → Write` 显式勾上。

> 坑 3：`HF_ENDPOINT` 镜像会污染端点，还会把网络故障伪装成令牌失效。如果这个环境变量被设成已失效的镜像（比如 `HF_ENDPOINT=https://hf-mirror.com`），它的失败（`SSL: UNEXPECTED_EOF_WHILE_READING` / 读超时）会被 `huggingface_hub` 报成 `Invalid user token`，看着像令牌坏了，其实是网络不通。正常应该指向 `https://huggingface.co`。另外 `SetEnvironmentVariable(...,'User')` 只写注册表，已经开着的进程仍是旧值，要新开终端才生效。端点还有第二个坑：`constants.ENDPOINT` 是导入时求值的，import 之后再改 `os.environ` 已经晚了（`HfApi` 也只有一个不带 endpoint 的构造），所以脚本在 `import huggingface_hub` 之前先扫一遍 `--endpoint`。这个预扫描是防御性的，端点若被改坏，加 `--endpoint https://huggingface.co` 仍能绕开。

> 坑 4：走代理时要关掉 `hf_transfer` 和 Xet。这两个都是 Rust 实现，不保证读取 `HTTP_PROXY`/`HTTPS_PROXY`，直连不通的机器上会卡死或失败。第一次传 4B 时开着 Xet，日志里 `cas::query_dedup` 持续报 `404 Not Found`（`cas-server.xethub.hf.co`），传到 `model-00001` 的 21% 就静默中断，仓库里只剩 README。加上 `HF_HUB_DISABLE_XET=1` 和 `HF_HUB_ENABLE_HF_TRANSFER=0` 改走经典 HTTPS 分片上传后就稳定跑完了。这两个变量脚本里都用 `setdefault` 设默认值，所以外部显式设成 `0` / `1` 就能覆盖。

> 坑 5：PowerShell 管道会把失败伪装成成功。`python ... | Tee-Object -FilePath x.log` 里 `$LASTEXITCODE` 取的是管道最后一个命令（`Tee-Object`）的退出码，于是 Python 中途挂掉也报 exit 0。第一次 4B 上传就是这样：任务显示"成功"，实际仓库里只有 README。要判断真实成败，用 `python ... *> x.log; "EXIT=$LASTEXITCODE"` 这种重定向写法，或者直接查仓库文件列表。

## 已知限制与调优建议

数据量偏小（23.7 万汉字 / 2.7 万样本），LoRA 是最合适的选择，全参微调容易过拟合。0.6B / 1.7B 这类小模型学文风反而更快更稳，建议先从小尺寸跑通再上 4B。

训练轮数上，小尺寸给了 4 epoch、大尺寸 3 epoch。若验证集 loss 抬头即过拟合，降低 `num_train_epochs` 或把 `learning_rate` 降到 5e-5。

14B 余量很小，若 OOM 请把 `cutoff_len` 降到 768 或把 `save_steps` 改小以减少峰值。

翻译任务占了样本的 54%。如果你更看重创作能力而不是互译，可以在 [build_dataset.py](data/build_dataset.py) 里给翻译样本加采样权重或减少翻译样本量。

想扩充数据的话，Steam 创意工坊的模组使用与本体完全相同的 JSON 结构，是继续加语料最自然的来源。把模组的 content 目录按 `<dir>/core` + `<dir>/loc_zh-hans` 的结构组织好，即可用 `extract.py --game-content <dir>` 继续扩大语料。

## 已实测验证的内容

本仓库的脚本与配置都真实跑通过一遍，不是只静态写出来的。简单记一下验证到哪一步：

- `extract.py` —— 全库 1170 个 JSON 文件全部解析成功（修复前有 25 个读不了），抽出 8691 条文本单元。
- `build_dataset.py` —— 生成 26,875 / 779 条训练/验证样本，任务分布与上文表格一致。
- `download_models.py` —— 已下载 `Qwen3-0.6B`（1.5GB）、`Qwen3-1.7B`（3.9GB）、`Qwen3-4B`（7.7GB）到 `models/`，中文路径正常。
- `train.ps1` + `configs/qwen3_0.6b_lora.yaml` —— 小样本冒烟训练跑通，并确认数据集与 `dataset_info.json` 被正确加载、loss 掩码正确（`label_ids` 里 prompt 部分全为 -100，只对 assistant 回复算损失）、`qwen3_nothink` 模板生效（不带 Qwen3 的思考段）、LoRA 可训练参数 10.09M / 606.14M = 1.67%、注意力使用 SDPA。
- `-Merge` —— LoRA 成功合并为完整模型（含 `chat_template.jinja`、`Modelfile`）。
- `infer/generate.py` —— 批量示例生成跑通。
- QLoRA 4bit —— `bitsandbytes 0.50.2` 下 0.6B 冒烟训练跑通，日志出现 `Quantizing model to 4 bit with bitsandbytes.`，8B/14B 的配置据此可用。
- `train_all.ps1` —— `-Sizes 0.6b,4b -Merge -DryRun` 通过；真跑时单个尺寸失败会正确打「0.6b 失败（退出码 1）」并继续跑后面的尺寸，最后汇总失败列表，不再是子进程变孤儿、脚本停在开始行。
- 4B 首跑失败并归因 —— 15:18 那次直接在 DataLoader 启动处崩：`_MultiProcessingDataLoaderIter.__init__ → w.start() → reduction.dump → storage._share_cuda_()` → `torch.AcceleratorError: CUDA error: out of memory`。当时同时满足两个不该同时出现的条件：一是 0.6B 还在训练，16 GB 显存已被占满，spawn 子进程建 CUDA 上下文时申请不到；二是 4B 配置还没开 `pure_bf16`，日志里是 `Upcasting trainable params to float32.`，LoRA 分支与整条激活链都是 fp32，显存需求翻倍。现在配置已加 `pure_bf16: true`，并且用 `train_all.ps1` 串行排队（一次只跑一个尺寸），不要再手动并发启动第二个尺寸。
- 中断续训 —— 0.6B 训到 step 1040 后手动停止，用 `overwrite_output_dir=false` 重启，日志确认从 checkpoint-1000 接上（step 1040 → 1120），loss 连续下降，步速 1.1 s/步。
- 0.6B 完整训练 —— 启动日志实测 `Num examples = 26,875` / `Instantaneous batch size per device = 4` / `Total train batch size = 16` / `Total optimization steps = 6,720`（1680 步/epoch × 4），步速约 1.0 s/步，loss 从 3.02 降到 1.8 上下，6720 步 / 1:22:03 跑完，末段 eval_loss 稳定在 2.406（6000/6200/6400/6600 步分别为 2.408/2.407/2.408/2.406，没有过拟合）。合并产物 `outputs\qwen3-0.6b-lora-merged`（1.13 GB 单文件）已用 CPU 跑通 8 条内置提示词。
- 4B 完整训练 —— `Pure bf16` 生效（日志 `Pure bf16 / BAdam detected, remaining trainable params in half precision.`，可训练参数 33,030,144 / 4,055,498,240 = 0.81%），`Total optimization steps = 5,040`（1680 步/epoch × 3）、`Instantaneous batch size = 1`，5040 步 / 6:00:21 跑完，末段 eval_loss 稳定在 1.949（4000→5000 步依次 1.938/1.948/1.949/1.946/1.948/1.949，已进入平台期）。训练期实测 GPU 占用中位数 94%、整卡功耗 111 W、步速 3.08 s/步，属于健康区间。训练期 `nvidia-smi` 显存实测 14,626 / 16,380 MiB（14.3 GB），余量约 1.7 GB，这证实了配置里「不能再加 per-device batch」的判断，改 `2×8` 必然 OOM。合并产物 `outputs\qwen3-4b-lora-merged`（7.51 GB / 3 分片）已在 GPU 上跑通同一组 8 条提示词。
- 0.6B vs 4B 同题对比（8 条内置提示词，两者各跑完整 8 条，均已复跑确认）—— 4B 全面更好。第 7 条中译英要求保留 `<b>`，4B 三组标记全部保留（`The light leaks through <b>cracks</b>. My mind is <b>clearer</b> than it's been <b>in any other time</b>.`），0.6B 标记全丢（`Light leaks through a Crack in the Skin. My brain is clearer than ever at any given time.`）。这一条正是区分点：数据集里中译英有两种指令，带标记的那种（`把下面的《密教模拟器》中文文本翻译成英文。保留 <b>、<i> 等富文本标记与换行结构。`）对应 22 条中英两侧都带标记的样本，属于真实训练过的能力，0.6B 没学会、4B 学会了。顺带修掉一个保真问题：`infer/generate.py` 原先把这条指令写成「…翻译成英文，保留…」，与数据集里的「…翻译成英文。保留…」差一个标点，已改为与训练指令逐字一致，两个尺寸也因此重新跑了一遍，上面用的是复跑结果。第 1 条卡牌描述，4B 是 `雪覆一切，日沉至地平线下。光线如冰霜般冷冽。`（克制、有画面感），0.6B 复跑为 `这里的每一块石头都深陷青苔，而石缝里那盆枯萎的多肉植物的根茎仍在挣扎。`，偏现代白话，不像原作语气。第 5 条结局正文，4B 结构完整、并正确保留了原文结尾的方括号附注，0.6B 则凭空混入了成就文本（`[恭喜您达成<b>爱情</b>之常规胜利结局。…]`），串了任务。第 2 条主题创作两者都偏短，4B 的意象更贴合「启」。结论是风格与格式遵循度上 4B > 0.6B，与 eval_loss（1.949 vs 2.406）一致。逐条原始输出见 `logs\verify_0.6b_merged.log`、`logs\verify_4b_merged.log`。
- `upload_hf.py` 上传 —— 两个仓库均已公开上传并核验文件清单与体积。0.6B 仓库 22 个文件 / 1.18 GB（合并模型 1,136.9 MB 单文件 + `lora/` 适配器 38.5 MB）；4B 仓库 25 个文件 / 7.65 GB（合并模型 3 分片 3,774.5 + 3,802.7 + 95.0 MB + `lora/` 适配器 126.1 MB）。0.6B 一次通过；4B 首跑因 Xet 经代理不稳而在 21% 静默中断（详见上文坑 4），关掉 Xet 后重跑约 25 分钟传完。
- 定位并修复两个让训练"看起来在跑、其实很慢或很瞎"的陷阱 —— 一是续训时 `per_device_train_batch_size` 被 checkpoint 的 `train_batch_size` 顶掉，导致等效 batch 从 16 变 32、总步数从 6720 变 3360、显存重新溢出（Shared 12 GB、11 s/步）；二是 `conda run` 不加 `--no-capture-output` 会把训练日志缓存到进程结束，整个训练过程中 `logs\train_*.log` 都是空的。两者都已修好并写进上文。

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
