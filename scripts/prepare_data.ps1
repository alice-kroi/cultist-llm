# 一键完成数据准备：从游戏目录抽取文本 → 构造训练数据集
#
# 用法（游戏装在默认位置可省略 -GameContent）：
#   .\scripts\prepare_data.ps1
#   .\scripts\prepare_data.ps1 -GameContent "X:\Games\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content"
#
param(
    [string]$GameContent = "<Steam 库>\steamapps\common\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content",
    [string]$EnvName = "llama-factory",
    [double]$EvalRatio = 0.03
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $Root "data"

if (-not (Test-Path $GameContent)) {
    throw "找不到游戏内容目录：$GameContent"
}

Write-Host "`n[1/2] 抽取游戏文本 ..." -ForegroundColor Cyan
& conda run -n $EnvName python (Join-Path $DataDir "extract.py") --game-content $GameContent
if ($LASTEXITCODE -ne 0) { throw "抽取失败" }

Write-Host "`n[2/2] 构造指令数据集 ..." -ForegroundColor Cyan
& conda run -n $EnvName python (Join-Path $DataDir "build_dataset.py") --eval-ratio $EvalRatio
if ($LASTEXITCODE -ne 0) { throw "数据集构建失败" }

Write-Host "`n数据已就绪：$(Join-Path $DataDir 'dataset')" -ForegroundColor Green
Write-Host "下一步： .\scripts\download_models.py --sizes 4b  然后  .\scripts\train.ps1 -Size 4b" -ForegroundColor Green