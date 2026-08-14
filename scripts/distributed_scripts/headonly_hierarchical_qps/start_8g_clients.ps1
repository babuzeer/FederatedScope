param(
    [string]$CaseDir = "scripts\distributed_scripts\headonly_hierarchical_qps\runs\headonly60c-1sub-paramqps-20260625",
    [string]$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe",
    [switch]$Restart,
    [switch]$LaunchAllClients,
    [switch]$SingleProcessClients,
    [int]$GroupCount = 8,
    [int]$TrainConcurrencyPerGroup = 8,
    [int]$MaxBatchesOverride = 1,
    [switch]$KeepClientLogs
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

$caseFull = (Resolve-Path $CaseDir).Path
$clientCfg = Join-Path $caseFull "configs\clients\client_000001.json"
if (-not (Test-Path $clientCfg)) {
    throw "client config not found: $clientCfg"
}

function Test-ClientCache {
    param([object]$Cfg)
    $cid = [int]$Cfg.client_id
    $flat0 = Join-Path $Cfg.cache_dir ("client_{0:D6}.pt" -f ($cid - 1))
    $flat1 = Join-Path $Cfg.cache_dir ("client_{0:D6}.pt" -f $cid)
    $nested = Join-Path $Cfg.cache_dir ("headonly_augmented\{0}\office-home_client_{1:D6}.pt" -f $Cfg.cache_version, $cid)
    return ((Test-Path $nested) -or (Test-Path $flat0) -or (Test-Path $flat1))
}

if (-not $LaunchAllClients) {
    $active = @()
    $missing = @()
    foreach ($file in Get-ChildItem (Join-Path $caseFull "configs\clients") -Filter "client_*.json") {
        $cfg = Get-Content $file.FullName -Raw | ConvertFrom-Json
        if (Test-ClientCache -Cfg $cfg) {
            $active += [int]$cfg.client_id
        } else {
            $missing += [int]$cfg.client_id
        }
    }
    if ($active.Count -le 0) {
        throw "no client feature cache found"
    }
    Write-Output ("active_cached_clients={0}" -f $active.Count)
    if ($missing.Count -gt 0) {
        Write-Output ("skipped_missing_cache_clients={0}" -f ($missing -join ","))
    }
}

if ($Restart) {
    .\scripts\distributed_scripts\headonly_hierarchical_qps\stop_clients.ps1 -CaseDir $caseFull -Force -ErrorAction SilentlyContinue
    if (-not $KeepClientLogs) {
        $manifestPath = Join-Path $caseFull "manifest.json"
        if (Test-Path $manifestPath) {
            $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
            $clientLogDir = $manifest.client_log_dir
            if (Test-Path $clientLogDir) {
                Remove-Item -Path (Join-Path $clientLogDir "client_*_events.jsonl") -Force -ErrorAction SilentlyContinue
                Remove-Item -Path (Join-Path $clientLogDir "client_*.log") -Force -ErrorAction SilentlyContinue
                Remove-Item -Path (Join-Path $clientLogDir "client_group_*_events.jsonl") -Force -ErrorAction SilentlyContinue
                Write-Output "cleared_client_log_dir=$clientLogDir"
            }
        }
        Remove-Item -Path (Join-Path $caseFull "system\client_*.stdout.log") -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $caseFull "system\client_*.stderr.log") -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $caseFull "system\client_group_*.stdout.log") -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $caseFull "system\client_group_*.stderr.log") -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $caseFull "system\client_pids.csv") -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $caseFull "system\client_group_pids.csv") -Force -ErrorAction SilentlyContinue
    }
}

if ($SingleProcessClients) {
    .\scripts\distributed_scripts\headonly_hierarchical_qps\launch_clients.ps1 `
        -CaseDir $caseFull `
        -PythonBin $PythonBin `
        -SkipMissingCache:(!$LaunchAllClients)
} else {
    .\scripts\distributed_scripts\headonly_hierarchical_qps\launch_client_groups.ps1 `
        -CaseDir $caseFull `
        -PythonBin $PythonBin `
        -GroupCount $GroupCount `
        -TrainConcurrencyPerGroup $TrainConcurrencyPerGroup `
        -MaxBatchesOverride $MaxBatchesOverride `
        -SkipMissingCache:(!$LaunchAllClients)
}

.\scripts\distributed_scripts\headonly_hierarchical_qps\status_clients.ps1 -CaseDir $caseFull
