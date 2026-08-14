param(
    [string]$RepoDir = "",
    [string]$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe",
    [string]$RunId = "",
    [string]$ExperimentSummary = "headonly60c-1sub-paramqps",
    [string]$SubserverAdvertiseHost = "10.112.81.135",
    [int]$RootPort = 58051,
    [int]$SubserverPort = 58061,
    [int]$ClientNum = 60,
    [int]$SubserverNum = 1,
    [int]$TotalRounds = 20,
    [double]$UploadTimeoutSec = 10.0,
    [double]$QpsWindowSec = 1.0,
    [string]$CacheDir = "exp/ggeur_headonly_real_cache/officehome_vitb16_60c_gen20_fcache_v1",
    [string]$CacheVersion = "officehome_vitb16_60c_gen20_fcache_v1",
    [string]$Device = "cpu",
    [int]$BatchSize = 64,
    [int]$LocalEpochs = 1,
    [int]$MaxBatches = 0,
    [double]$TrainTimeoutSec = 0.0,
    [switch]$CheckCache
)

$ErrorActionPreference = "Stop"

if (-not $RepoDir) {
    $RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}
if (-not (Test-Path $PythonBin) -and -not (Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
    throw "PythonBin not found: $PythonBin"
}

Set-Location $RepoDir

if ([System.IO.Path]::IsPathRooted($CacheDir)) {
    $cacheFull = $CacheDir
} else {
    $cacheFull = [System.IO.Path]::GetFullPath((Join-Path $RepoDir $CacheDir))
}

$sampleNested = Join-Path $cacheFull ("headonly_augmented\{0}\office-home_client_000001.pt" -f $CacheVersion)
$sampleFlat = Join-Path $cacheFull "client_000001.pt"
if ($CheckCache -and -not ((Test-Path $sampleNested) -or (Test-Path $sampleFlat))) {
    throw "feature cache not found. Tried: $sampleNested and $sampleFlat"
}
if (-not ((Test-Path $sampleNested) -or (Test-Path $sampleFlat))) {
    Write-Warning "feature cache sample was not found yet. Case will be generated, but clients require cache before launch."
    Write-Warning "tried: $sampleNested"
    Write-Warning "tried: $sampleFlat"
}

$argsList = @(
    "scripts/distributed_scripts/headonly_hierarchical_qps/prepare_case.py",
    "--experiment-summary", $ExperimentSummary,
    "--client-num", "$ClientNum",
    "--subserver-num", "$SubserverNum",
    "--total-rounds", "$TotalRounds",
    "--root-host", "127.0.0.1",
    "--root-port", "$RootPort",
    "--subserver-advertise-host", $SubserverAdvertiseHost,
    "--subserver-port", "$SubserverPort",
    "--upload-timeout-sec", "$UploadTimeoutSec",
    "--qps-window-sec", "$QpsWindowSec",
    "--cache-dir", $cacheFull,
    "--cache-version", $CacheVersion,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--local-epochs", "$LocalEpochs",
    "--max-batches", "$MaxBatches",
    "--train-timeout-sec", "$TrainTimeoutSec"
)
if ($RunId) {
    $argsList += @("--run-id", $RunId)
}

$casePath = (& $PythonBin @argsList | Select-Object -Last 1).Trim()
$caseFull = (Resolve-Path $casePath).Path
Write-Output "case_dir=$caseFull"
