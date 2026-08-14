param(
    [string]$Remote = "root@10.112.81.135",
    [string]$RemoteRoot = "/root/autodl-tmp",
    [string]$SourceRoot = "D:\Projects\FederatedScope",
    [string]$ModelPath = "D:\Projects\2025CVPR_GGEUR\models\open_clip_vitb16.bin",
    [string]$DatasetPath = "D:\Projects\FederatedScope\OfficeHomeDataset_10072016",
    [switch]$SkipDataset,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"

function Run-Step($Title, [scriptblock]$Body) {
    Write-Host ""
    Write-Host "===== $Title ====="
    & $Body
}

if (-not (Test-Path $SourceRoot)) {
    throw "SourceRoot does not exist: $SourceRoot"
}
if (-not $SkipModel -and -not (Test-Path $ModelPath)) {
    throw "ModelPath does not exist: $ModelPath"
}
if (-not $SkipDataset -and -not (Test-Path $DatasetPath)) {
    throw "DatasetPath does not exist: $DatasetPath"
}

$PackageScript = Join-Path $SourceRoot "scripts\package_headonly_two_machine_bundle.ps1"
if (-not (Test-Path $PackageScript)) {
    throw "Missing package script: $PackageScript"
}

$ZipPath = Join-Path $env:TEMP "FederatedScope_headonly_two_machine.zip"

Run-Step "Check SSH" {
    ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $Remote "hostname; date; mkdir -p '$RemoteRoot'"
}

Run-Step "Package code" {
    powershell -ExecutionPolicy Bypass -File $PackageScript -SourceRoot $SourceRoot -OutputZip $ZipPath
    Get-Item $ZipPath | Select-Object FullName,Length,LastWriteTime
}

Run-Step "Prepare remote directories" {
    ssh $Remote "mkdir -p '$RemoteRoot/FederatedScope' '$RemoteRoot/models' '$RemoteRoot/datasets'"
}

Run-Step "Upload code package" {
    scp $ZipPath "${Remote}:$RemoteRoot/FederatedScope_headonly_two_machine.zip"
}

Run-Step "Unpack code on remote" {
    ssh $Remote "cd '$RemoteRoot/FederatedScope' && (unzip -o '$RemoteRoot/FederatedScope_headonly_two_machine.zip' >/tmp/headonly_unzip.log 2>&1 || python3 - <<'PY'
import zipfile
zipfile.ZipFile('$RemoteRoot/FederatedScope_headonly_two_machine.zip').extractall('$RemoteRoot/FederatedScope')
PY
) && python3 -m py_compile scripts/benchmark_headonly_mlp_download_window.py scripts/summarize_headonly_qps_runs.py"
}

if (-not $SkipModel) {
    Run-Step "Upload CLIP model" {
        scp $ModelPath "${Remote}:$RemoteRoot/models/open_clip_vitb16.bin"
    }
}

if (-not $SkipDataset) {
    Run-Step "Upload OfficeHome dataset" {
        scp -r $DatasetPath "${Remote}:$RemoteRoot/datasets/"
    }
}

Run-Step "Verify remote layout" {
    ssh $Remote "set -e
echo '[code]'
ls -ld '$RemoteRoot/FederatedScope'
ls '$RemoteRoot/FederatedScope/scripts/distributed_scripts/ggeur_headonly_download_qps'
echo '[model]'
ls -lh '$RemoteRoot/models/open_clip_vitb16.bin' || true
echo '[dataset]'
find '$RemoteRoot/datasets/OfficeHomeDataset_10072016' -maxdepth 1 -mindepth 1 -printf '%f\n' 2>/dev/null | sort || true
echo '[python]'
python3 --version
"
}

Write-Host ""
Write-Host "Deployment finished."
Write-Host "Remote repo: $RemoteRoot/FederatedScope"
Write-Host "Remote model: $RemoteRoot/models/open_clip_vitb16.bin"
Write-Host "Remote dataset: $RemoteRoot/datasets/OfficeHomeDataset_10072016"
