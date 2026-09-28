# 《密教模拟器》文本模型 —— 多尺寸串行训练脚本
#
# 依次训练若干尺寸，每个尺寸默认训练完就合并成完整模型，日志分别写到 logs\ 下。
# 因为 16GB 显存一次只能跑一个尺寸，所以这里是串行排队而不是并行。
#
# 示例（-Sizes 用逗号分隔，注意不要用空格分隔）：
#   .\scripts\train_all.ps1 -Sizes 0.6b,1.7b,4b -Merge
#   .\scripts\train_all.ps1 -Sizes 8b,14b -Merge -ModelDir X:\cultist-llm-models
#   .\scripts\train_all.ps1 -Sizes 4b -Merge -SkipExisting      # 已有合并结果就跳过
#
# 说明：本脚本只是按顺序调用 train.ps1，所有实际逻辑都在 train.ps1 里。
param(
    # 这里不用 ValidateSet：powershell -File 传 "0.6b,1.7b,4b" 时会被当成单个字符串，
    # ValidateSet 会在下面的逗号拆分之前就报错。改为拆完再手工校验。
    [string[]]$Sizes = @("0.6b", "1.7b", "4b"),

    [string]$ModelDir = "",
    [string]$EnvName = "llama-factory",
    [switch]$Merge,
    [switch]$SkipExisting,
    [switch]$DryRun,
    [string]$Extra = ""
)

$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $PSScriptRoot
$Trainer = Join-Path $PSScriptRoot "train.ps1"
$LogDir  = Join-Path $Root "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

# powershell -File 方式传参时数组可能只绑到第一个值，这里兜底把逗号串拆开
if ($Sizes.Count -eq 1 -and $Sizes[0] -like "*,*") {
    $Sizes = @($Sizes[0].Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

$ValidSizes = @("0.6b", "1.7b", "4b", "8b", "14b")
$Unknown = @($Sizes | Where-Object { $ValidSizes -notcontains $_ })
if ($Unknown.Count -gt 0) {
    throw "未知尺寸：$($Unknown -join ', ')（可选：$($ValidSizes -join ', ')）"
}

# 合并后的模型目录名，与 train.ps1 的 $FinalOut-merged 保持一致
$MergedNames = @{
    "0.6b" = "qwen3-0.6b-lora-merged"
    "1.7b" = "qwen3-1.7b-lora-merged"
    "4b"   = "qwen3-4b-lora-merged"
    "8b"   = "qwen3-8b-qlora-merged"
    "14b"  = "qwen3-14b-qlora-merged"
}

Write-Host "计划训练：$($Sizes -join ' -> ')" -ForegroundColor Cyan
if ($Merge) { Write-Host "训练后合并 LoRA" -ForegroundColor Cyan }
if ($DryRun) { Write-Host "[DryRun] 只打印，不实际执行" -ForegroundColor Yellow }

$Failed = @()
foreach ($Size in $Sizes) {
    $Merged = Join-Path (Join-Path $Root "outputs") $MergedNames[$Size]

    if ($SkipExisting -and (Test-Path (Join-Path $Merged "config.json"))) {
        Write-Host "`n[跳过] $Size 已有合并模型：$Merged" -ForegroundColor Yellow
        continue
    }

    $TrainArgs = @("-ExecutionPolicy", "Bypass", "-File", $Trainer, "-Size", $Size, "-EnvName", $EnvName)
    if (-not [string]::IsNullOrWhiteSpace($ModelDir)) { $TrainArgs += @("-ModelDir", $ModelDir) }
    if (-not [string]::IsNullOrWhiteSpace($Extra))    { $TrainArgs += @("-Extra", $Extra) }
    if ($Merge) { $TrainArgs += "-Merge" }
    if ($DryRun) { $TrainArgs += "-DryRun" }

    $LogFile = Join-Path $LogDir "train_$Size.log"
    Write-Host "`n================ $Size 开始 ================" -ForegroundColor Green
    Write-Host "日志：$LogFile"

    # 注意（踩过的坑）：PowerShell 5.1 下 `*>` 会把子进程写到 stderr 的内容（例如 python
    # 的 UserWarning，jieba 加载时就会打一条 pkg_resources 弃用警告）转成 ErrorRecord，
    # 而本脚本开头设了 $ErrorActionPreference = "Stop"，于是这些**警告**会被当成致命错误，
    # 把 train_all.ps1 本身干掉，子进程则变成孤儿继续在后台跑（表现为本脚本日志停在
    # "xxx 开始" 就不动了）。所以调用子进程期间临时放宽为 Continue，只按退出码判断成败。
    $PrevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($DryRun) {
            & powershell @TrainArgs
        } else {
            & powershell @TrainArgs *> $LogFile
        }
        $Code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PrevEap
    }
    if ($Code -eq 0) {
        Write-Host "================ $Size 完成 ================" -ForegroundColor Green
    } else {
        Write-Host "================ $Size 失败（退出码 $Code） ================" -ForegroundColor Red
        $Failed += $Size
        # 继续跑后面的尺寸；单个尺寸失败不该拖垮整批
    }
}

Write-Host ""
if ($Failed.Count -gt 0) {
    Write-Host "以下尺寸失败：$($Failed -join ', ')" -ForegroundColor Red
    Write-Host "逐个排查：Get-Content logs\train_<size>.log -Tail 40"
    exit 1
}
Write-Host "全部完成。合并后的模型在 outputs\ 下，可用 infer\generate.py 生成。验收：" -ForegroundColor Green
foreach ($Size in $Sizes) {
    Write-Host ("  " + (Join-Path (Join-Path $Root "outputs") $MergedNames[$Size]))
}