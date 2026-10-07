# 《密教模拟器》文本模型 —— 一键训练脚本
#
# 示例：
#   .\scripts\train.ps1 -Size 4b                                    # 训练 Qwen3-4B LoRA
#   .\scripts\train.ps1 -Size 8b                                    # 训练 Qwen3-8B QLoRA
#   .\scripts\train.ps1 -Size 4b -ModelDir X:\models\cultist-llm     # 权重在别的盘
#   .\scripts\train.ps1 -Size 4b -Merge                             # 训练完合并成完整模型
#   .\scripts\train.ps1 -Size 4b -Extra "num_train_epochs=5 learning_rate=5e-5"
#
# 实现说明：不修改配置文件内容，而是用 LLaMA-Factory 原生的 OmegaConf 覆盖语法
# （key=value，注意不是 --key value）追加参数，避免改写 YAML 带来的编码问题。
#
param(
    [ValidateSet("0.6b", "1.7b", "4b", "8b", "14b")]
    [string]$Size = "4b",

    [string]$Config = "",           # 留空则按 -Size 自动选择
    [string]$ModelDir = "",         # 覆盖配置里的模型根目录
    [string]$OutputDir = "",        # 覆盖输出目录
    [string]$EnvName = "llama-factory",
    [switch]$Merge,                 # 训练后合并 LoRA 权重
    [switch]$DryRun,                # 只打印命令不执行
    [string]$Extra = ""             # 追加的覆盖参数，形如 key=value，空格分隔
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot          # cultist-llm 根目录

# ---------- 选择配置文件 ----------
$Auto = @{
    "0.6b" = "qwen3_0.6b_lora.yaml"
    "1.7b" = "qwen3_1.7b_lora.yaml"
    "4b"   = "qwen3_4b_lora.yaml"
    "8b"   = "qwen3_8b_qlora.yaml"
    "14b"  = "qwen3_14b_qlora.yaml"
}
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $Root ("configs\" + $Auto[$Size])
} elseif (-not [System.IO.Path]::IsPathRooted($Config)) {
    $Config = Join-Path $Root $Config
}
if (-not (Test-Path $Config)) { throw "配置文件不存在：$Config" }
Write-Host "配置文件 : $Config" -ForegroundColor Cyan

# ---------- 检查数据集 ----------
$DatasetDir = Join-Path $Root "data\dataset"
if (-not (Test-Path (Join-Path $DatasetDir "dataset_info.json"))) {
    throw "未找到数据集，请先运行： .\scripts\prepare_data.ps1"
}

# ---------- 读取配置里的模型与输出路径（显式 UTF8，避免中文注释被按 ANSI 解码） ----------
$YamlText  = Get-Content -Raw -Encoding UTF8 $Config
$ModelPath = ([regex]::Match($YamlText, '(?m)^model_name_or_path:\s*(.+?)\s*$')).Groups[1].Value
$OutPath   = ([regex]::Match($YamlText, '(?m)^output_dir:\s*(.+?)\s*$')).Groups[1].Value
if ([string]::IsNullOrWhiteSpace($ModelPath)) { throw "配置里读不到 model_name_or_path：$Config" }
# 配置里的路径都是相对仓库根的（这样 clone 到任何目录都能跑），这里解析成绝对路径，
# 后面就与当前工作目录无关了
if (-not [System.IO.Path]::IsPathRooted($ModelPath)) { $ModelPath = Join-Path $Root $ModelPath }
if (-not [string]::IsNullOrWhiteSpace($OutPath) -and -not [System.IO.Path]::IsPathRooted($OutPath)) {
    $OutPath = Join-Path $Root $OutPath
}
Write-Host "基座模型 : $ModelPath"
Write-Host "输出目录 : $OutPath"

# ---------- 组装覆盖参数 ----------
$Overrides = @()
if (-not [string]::IsNullOrWhiteSpace($ModelDir)) {
    $NewRoot = [System.IO.Path]::GetFullPath($ModelDir)
    $Result  = Join-Path $NewRoot (Split-Path -Leaf $ModelPath)
    $Overrides += "model_name_or_path=$Result"
    Write-Host "模型路径已改为：$Result" -ForegroundColor Yellow
}
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $Result = [System.IO.Path]::GetFullPath($OutputDir)
    $Overrides += "output_dir=$Result"
    Write-Host "输出目录已改为：$Result" -ForegroundColor Yellow
}
if (-not [string]::IsNullOrWhiteSpace($Extra)) {
    $Overrides += $Extra.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}

# ---------- QLoRA 依赖检查 ----------
if ($YamlText -match "quantization_bit:\s*4") {
    # 同上：PS 5.1 下对原生命令做 `2>$null` 会生成 ErrorRecord，配上 ErrorActionPreference=Stop
    # 会把 python 的普通警告误判成致命错误，这里临时放宽
    $PrevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $bnb = & conda run -n $EnvName python -c "import bitsandbytes; print('ok')" 2>$null }
    finally { $ErrorActionPreference = $PrevEap }
    if ($bnb -notmatch "ok") {
        Write-Host "[错误] 该配置使用 4bit 量化，但 $EnvName 环境缺少 bitsandbytes。" -ForegroundColor Red
        Write-Host "       请先执行： conda run -n $EnvName pip install -U bitsandbytes" -ForegroundColor Red
        throw "缺少 bitsandbytes"
    }
}

# ---------- 训练 ----------
# 必须加 --no-capture-output（等价 --live-stream）。不加的话 `conda run` 会把子进程的
# stdout/stderr 全部缓存起来、等进程结束才吐出来，于是日志文件在训练的几个小时里
# 一直是空的（踩过：logs\train_0.6b.log 只有几百字节，出问题完全看不到 traceback）。
$ArgList = @("run", "--no-capture-output", "-n", $EnvName, "llamafactory-cli", "train", $Config) + $Overrides
Write-Host "`n执行： conda $($ArgList -join ' ')" -ForegroundColor Cyan
if ($DryRun) { exit 0 }

# 切到仓库根目录再跑：配置里的 dataset_dir（data\dataset）是相对仓库根的
Push-Location $Root
try { & conda @ArgList } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw "训练失败（退出码 $LASTEXITCODE）" }

# ---------- 训练指标图表 ----------
# 从 trainer_state.json / trainer_log.jsonl / all_results.json 生成多面板指标图，
# 便于训练后的效果分析与后续开发（train/eval loss、学习率、梯度范数、每步耗时）
$FinalOut = if ($Overrides -match "^output_dir=") {
    ($Overrides | Where-Object { $_ -like "output_dir=*" }) -replace "^output_dir=", ""
} else { $OutPath }

if ($FinalOut -and (Test-Path (Join-Path $FinalOut "trainer_state.json"))) {
    Write-Host "`n生成训练指标图表 ..." -ForegroundColor Cyan
    & conda run --no-capture-output -n $EnvName python (Join-Path $Root "scripts\plot_training.py") --output-dir $FinalOut
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[警告] 指标图表生成失败（不影响训练结果）" -ForegroundColor Yellow
    }
}

# ---------- 可选：合并 LoRA 权重 ----------
if ($Merge) {
    $MergedDir = "$FinalOut-merged"
    Write-Host "`n合并 LoRA 到完整模型：$MergedDir" -ForegroundColor Cyan
    & conda run --no-capture-output -n $EnvName llamafactory-cli export `
        --model_name_or_path $ModelPath `
        --adapter_name_or_path $FinalOut `
        --template qwen3_nothink `
        --finetuning_type lora `
        --export_dir $MergedDir `
        --export_size 4 `
        --export_legacy_format false
    if ($LASTEXITCODE -ne 0) { throw "合并失败（退出码 $LASTEXITCODE）" }
    Write-Host "合并完成：$MergedDir" -ForegroundColor Green
}

Write-Host "`n训练完成。" -ForegroundColor Green
Write-Host "  权重      : $FinalOut"
Write-Host "  试试效果  : conda run -n $EnvName python infer/generate.py --model `"$ModelPath`" --adapter `"$FinalOut`""
Write-Host "  若要对比多个尺寸：分别加 -Merge 合并后，用 infer/generate.py 跑同一组提示词。"