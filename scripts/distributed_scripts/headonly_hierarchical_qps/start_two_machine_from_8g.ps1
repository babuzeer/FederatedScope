param(
    [string]$CaseDir = "scripts\distributed_scripts\headonly_hierarchical_qps\runs\headonly60c-1sub-paramqps-20260625",
    [string]$ServerHost = "10.112.81.135",
    [string]$ServerUser = "root",
    [string]$ServerRepo = "/root/autodl-tmp/FederatedScope",
    [string]$ServerPython = "/root/.local/share/mamba/envs/GGEUR/bin/python",
    [string]$ClientPython = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe",
    [switch]$Restart,
    [switch]$SkipUpload,
    [switch]$LaunchAllClients
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $RepoRoot

$caseFull = (Resolve-Path $CaseDir).Path
$runId = Split-Path $caseFull -Leaf
$bundle = Join-Path $caseFull ("server_bundle_{0}.zip" -f $runId)
if (-not (Test-Path $bundle)) {
    throw "server bundle not found: $bundle"
}

function Test-ClientCache {
    param([object]$Cfg)
    $cid = [int]$Cfg.client_id
    $flat0 = Join-Path $Cfg.cache_dir ("client_{0:D6}.pt" -f ($cid - 1))
    $flat1 = Join-Path $Cfg.cache_dir ("client_{0:D6}.pt" -f $cid)
    $nested = Join-Path $Cfg.cache_dir ("headonly_augmented\{0}\office-home_client_{1:D6}.pt" -f $Cfg.cache_version, $cid)
    return ((Test-Path $nested) -or (Test-Path $flat0) -or (Test-Path $flat1))
}

$clientConfigFiles = Get-ChildItem (Join-Path $caseFull "configs\clients") -Filter "client_*.json"
$active = @()
$missing = @()
foreach ($file in $clientConfigFiles) {
    $cfg = Get-Content $file.FullName -Raw | ConvertFrom-Json
    if ($LaunchAllClients -or (Test-ClientCache -Cfg $cfg)) {
        $active += [int]$cfg.client_id
    } else {
        $missing += [int]$cfg.client_id
    }
}
if ($active.Count -le 0) {
    throw "no active clients; no client feature cache found"
}
Write-Output ("active_clients={0}" -f $active.Count)
if ($missing.Count -gt 0) {
    Write-Output ("skipped_missing_cache_clients={0}" -f ($missing -join ","))
}

$subCfgPath = Join-Path $caseFull "configs\subserver_1.json"
$subCfg = Get-Content $subCfgPath -Raw | ConvertFrom-Json
$subCfg.expected_clients = $active.Count
$subCfg | ConvertTo-Json -Depth 20 | Set-Content -Path $subCfgPath -Encoding UTF8

& (Join-Path $ScriptDir "package_case_for_4090.ps1") -CaseDir $caseFull | Write-Output

$remote = "$ServerUser@$ServerHost"
$remoteBundle = "$ServerRepo/$(Split-Path $bundle -Leaf)"
$remoteCase = "scripts/distributed_scripts/headonly_hierarchical_qps/runs/$runId"

if (-not $SkipUpload) {
    scp "$bundle" "${remote}:${ServerRepo}/"
    if ($LASTEXITCODE -ne 0) {
        throw "scp to $remote failed with exit code $LASTEXITCODE"
    }
}

$restartFlag = if ($Restart) { "1" } else { "0" }
$remoteCmd = @"
set -e
cd "$ServerRepo"
if [ "$restartFlag" = "1" ] && [ -x scripts/distributed_scripts/headonly_hierarchical_qps/stop_case.sh ]; then
  bash scripts/distributed_scripts/headonly_hierarchical_qps/stop_case.sh "$remoteCase" || true
fi
rm -rf scripts/distributed_scripts/headonly_hierarchical_qps
"$ServerPython" -m zipfile -e "$(Split-Path $bundle -Leaf)" .
test -f scripts/distributed_scripts/headonly_hierarchical_qps/start_4090.sh
export PYTHON_BIN="$ServerPython"
export CASE_DIR="$remoteCase"
export RESTART="$restartFlag"
bash scripts/distributed_scripts/headonly_hierarchical_qps/start_4090.sh
"@

$remoteCmdLf = $remoteCmd -replace "`r`n", "`n"
$remoteCmdLf | ssh $remote "bash -s"
if ($LASTEXITCODE -ne 0) {
    throw "remote 4090 startup failed with exit code $LASTEXITCODE"
}

.\scripts\distributed_scripts\headonly_hierarchical_qps\start_8g_clients.ps1 `
    -CaseDir $caseFull `
    -PythonBin $ClientPython `
    -Restart:$Restart `
    -LaunchAllClients:$LaunchAllClients
