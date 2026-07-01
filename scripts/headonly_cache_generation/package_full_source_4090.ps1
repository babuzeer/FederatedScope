param(
    [string]$OutputZip = "scripts\headonly_cache_generation\federatedscope_full_source_4090_bundle.zip"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$OutputPath = Join-Path $RepoRoot $OutputZip
$OutputDir = Split-Path -Parent $OutputPath
$Stage = Join-Path $ScriptDir ".full_source_stage"

if (Test-Path -LiteralPath $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Dirs = @(
    "federatedscope"
)

foreach ($Relative in $Dirs) {
    $Source = Join-Path $RepoRoot $Relative
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing source directory: $Relative"
    }
    $Target = Join-Path $Stage $Relative
    Copy-Item -LiteralPath $Source -Destination $Target -Recurse -Force
}

$Files = @(
    "scripts\parse_headonly_system_metrics.py",
    "scripts\example_configs\ggeur_headonly_system\officehome_lds_vit_headonly_fedavg_gen20.yaml",
    "scripts\headonly_cache_generation\officehome_vit_cachegen_gen20.yaml",
    "scripts\headonly_cache_generation\officehome_vit_cachegen_400c_random25_gen20.yaml",
    "scripts\headonly_cache_generation\officehome_vit_cachegen_400c_random25_gen20_8g.yaml",
    "scripts\headonly_cache_generation\run_officehome_vit_cachegen_4090.sh",
    "scripts\headonly_cache_generation\README.md"
)

foreach ($Relative in $Files) {
    $Source = Join-Path $RepoRoot $Relative
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Missing bundle file: $Relative"
    }
    $Target = Join-Path $Stage $Relative
    $TargetDir = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target -Force
}

if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $OutputPath -Force
Remove-Item -LiteralPath $Stage -Recurse -Force

Write-Host "bundle=$OutputPath"
