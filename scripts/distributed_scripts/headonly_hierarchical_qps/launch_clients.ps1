param(
    [Parameter(Mandatory = $true)][string]$CaseDir,
    [string]$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe",
    [int]$StartClient = 1,
    [int]$EndClient = 0,
    [double]$StartGapSec = 0.05,
    [switch]$SkipMissingCache,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
$CaseDir = (Resolve-Path $CaseDir).Path
$ClientConfigDir = Join-Path $CaseDir "configs\clients"
$SystemDir = Join-Path $CaseDir "system"
$WorkerScript = Join-Path $ScriptDir "headonly_hierarchical_qps.py"

if (-not (Test-Path $PythonBin) -and -not (Get-Command $PythonBin -ErrorAction SilentlyContinue)) {
    throw "PythonBin not found: $PythonBin"
}
if (-not (Test-Path $ClientConfigDir)) {
    throw "client config dir not found: $ClientConfigDir"
}

New-Item -ItemType Directory -Force -Path $SystemDir | Out-Null

if ($EndClient -le 0) {
    $EndClient = (Get-ChildItem $ClientConfigDir -Filter "client_*.json" |
        Measure-Object).Count
}

$rows = @()
for ($clientId = $StartClient; $clientId -le $EndClient; $clientId++) {
    $config = Join-Path $ClientConfigDir ("client_{0:D6}.json" -f $clientId)
    if (-not (Test-Path $config)) {
        throw "client config not found: $config"
    }
    if ($SkipMissingCache) {
        $cfg = Get-Content $config -Raw | ConvertFrom-Json
        $cid = [int]$cfg.client_id
        $flat0 = Join-Path $cfg.cache_dir ("client_{0:D6}.pt" -f ($cid - 1))
        $flat1 = Join-Path $cfg.cache_dir ("client_{0:D6}.pt" -f $cid)
        $nested = Join-Path $cfg.cache_dir ("headonly_augmented\{0}\office-home_client_{1:D6}.pt" -f $cfg.cache_version, $cid)
        if (-not ((Test-Path $nested) -or (Test-Path $flat0) -or (Test-Path $flat1))) {
            Write-Output ("skip_client_no_cache={0}" -f $clientId)
            continue
        }
    }
    $stdout = Join-Path $SystemDir ("client_{0:D6}.stdout.log" -f $clientId)
    $stderr = Join-Path $SystemDir ("client_{0:D6}.stderr.log" -f $clientId)

    if ($Foreground) {
        & $PythonBin $WorkerScript client --config $config
        $clientProcessId = 0
    } else {
        $argLine = ('"{0}" client --config "{1}"' -f $WorkerScript, $config)
        $proc = Start-Process -FilePath $PythonBin `
            -ArgumentList $argLine `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $clientProcessId = $proc.Id
    }

    $rows += [pscustomobject]@{
        client_id = $clientId
        pid = $clientProcessId
        config = $config
        stdout = $stdout
        stderr = $stderr
    }
    Start-Sleep -Milliseconds ([int]($StartGapSec * 1000))
}

$pidFile = Join-Path $SystemDir "client_pids.csv"
$rows | Export-Csv -Path $pidFile -NoTypeInformation -Encoding ASCII
Write-Output "client_pid_file=$pidFile"
Write-Output "started_clients=$($rows.Count)"
