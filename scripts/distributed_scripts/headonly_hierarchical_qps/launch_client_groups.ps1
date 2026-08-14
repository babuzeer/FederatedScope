param(
    [Parameter(Mandatory = $true)][string]$CaseDir,
    [string]$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe",
    [int]$StartClient = 1,
    [int]$EndClient = 0,
    [int]$GroupCount = 8,
    [int]$TrainConcurrencyPerGroup = 8,
    [int]$MaxBatchesOverride = 1,
    [double]$TrainTimeoutOverride = -1,
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
if ($GroupCount -le 0) {
    throw "GroupCount must be positive"
}
if ($TrainConcurrencyPerGroup -le 0) {
    throw "TrainConcurrencyPerGroup must be positive"
}

New-Item -ItemType Directory -Force -Path $SystemDir | Out-Null

if ($EndClient -le 0) {
    $EndClient = (Get-ChildItem $ClientConfigDir -Filter "client_*.json" | Measure-Object).Count
}

function Test-ClientCache {
    param([object]$Cfg)
    $cid = [int]$Cfg.client_id
    $flat0 = Join-Path $Cfg.cache_dir ("client_{0:D6}.pt" -f ($cid - 1))
    $flat1 = Join-Path $Cfg.cache_dir ("client_{0:D6}.pt" -f $cid)
    $nested = Join-Path $Cfg.cache_dir ("headonly_augmented\{0}\office-home_client_{1:D6}.pt" -f $Cfg.cache_version, $cid)
    return ((Test-Path $nested) -or (Test-Path $flat0) -or (Test-Path $flat1))
}

$clientIds = @()
for ($clientId = $StartClient; $clientId -le $EndClient; $clientId++) {
    $config = Join-Path $ClientConfigDir ("client_{0:D6}.json" -f $clientId)
    if (-not (Test-Path $config)) {
        throw "client config not found: $config"
    }
    if ($SkipMissingCache) {
        $cfg = Get-Content $config -Raw | ConvertFrom-Json
        if (-not (Test-ClientCache -Cfg $cfg)) {
            Write-Output ("skip_client_no_cache={0}" -f $clientId)
            continue
        }
    }
    $clientIds += $clientId
}

if ($clientIds.Count -le 0) {
    throw "no clients selected"
}

$groupSize = [int][Math]::Ceiling($clientIds.Count / [double]$GroupCount)
$rows = @()
$groupId = 0
for ($offset = 0; $offset -lt $clientIds.Count; $offset += $groupSize) {
    $groupId += 1
    $lastOffset = [Math]::Min($offset + $groupSize - 1, $clientIds.Count - 1)
    $range = $clientIds[$offset..$lastOffset]
    $rangeStart = [int]$range[0]
    $rangeEnd = [int]$range[-1]

    $stdout = Join-Path $SystemDir ("client_group_{0:D3}.stdout.log" -f $groupId)
    $stderr = Join-Path $SystemDir ("client_group_{0:D3}.stderr.log" -f $groupId)

    $argsList = @(
        ('"{0}"' -f $WorkerScript),
        "client-group",
        "--config-dir", ('"{0}"' -f $ClientConfigDir),
        "--start-client", $rangeStart,
        "--end-client", $rangeEnd,
        "--group-id", $groupId,
        "--train-concurrency", $TrainConcurrencyPerGroup,
        "--max-batches-override", $MaxBatchesOverride
    )
    if ($TrainTimeoutOverride -ge 0) {
        $argsList += @("--train-timeout-override", $TrainTimeoutOverride)
    }
    $argLine = $argsList -join " "

    if ($Foreground) {
        & $PythonBin $WorkerScript client-group `
            --config-dir $ClientConfigDir `
            --start-client $rangeStart `
            --end-client $rangeEnd `
            --group-id $groupId `
            --train-concurrency $TrainConcurrencyPerGroup `
            --max-batches-override $MaxBatchesOverride
        $groupProcessId = 0
    } else {
        $proc = Start-Process -FilePath $PythonBin `
            -ArgumentList $argLine `
            -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
        $groupProcessId = $proc.Id
    }

    $rows += [pscustomobject]@{
        group_id = $groupId
        pid = $groupProcessId
        start_client = $rangeStart
        end_client = $rangeEnd
        client_count = $range.Count
        stdout = $stdout
        stderr = $stderr
    }
}

$pidFile = Join-Path $SystemDir "client_group_pids.csv"
$rows | Export-Csv -Path $pidFile -NoTypeInformation -Encoding ASCII
Write-Output "client_group_pid_file=$pidFile"
Write-Output "started_groups=$($rows.Count)"
Write-Output "covered_clients=$($clientIds.Count)"
Write-Output "train_concurrency_per_group=$TrainConcurrencyPerGroup"
Write-Output "max_batches_override=$MaxBatchesOverride"
