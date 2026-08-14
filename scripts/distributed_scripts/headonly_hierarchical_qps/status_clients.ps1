param(
    [Parameter(Mandatory = $true)][string]$CaseDir
)

$ErrorActionPreference = "Stop"

$CaseDir = (Resolve-Path $CaseDir).Path
$SystemDir = Join-Path $CaseDir "system"
$PidFile = Join-Path $SystemDir "client_pids.csv"
$GroupPidFile = Join-Path $SystemDir "client_group_pids.csv"
$ManifestPath = Join-Path $CaseDir "manifest.json"

$rows = @()
if (Test-Path $PidFile) {
    $rows = Import-Csv $PidFile
}
$groupRows = @()
if (Test-Path $GroupPidFile) {
    $groupRows = Import-Csv $GroupPidFile
}
$running = 0
foreach ($row in $rows) {
    $pidValue = [int]$row.pid
    if ($pidValue -le 0) {
        continue
    }
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        $running += 1
    }
}
$runningGroups = 0
foreach ($row in $groupRows) {
    $pidValue = [int]$row.pid
    if ($pidValue -le 0) {
        continue
    }
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        $runningGroups += 1
    }
}

$finished = 0
$uploads = 0
$paramReads = 0
$eventFiles = 0
$clientTotal = $rows.Count
if (Test-Path $ManifestPath) {
    $manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    if ($manifest.client_num) {
        $clientTotal = [int]$manifest.client_num
    }
    $clientLogDir = $manifest.client_log_dir
    if (Test-Path $clientLogDir) {
        $eventFileItems = @(Get-ChildItem $clientLogDir -Filter "client_*_events.jsonl" -File -ErrorAction SilentlyContinue)
        $eventFiles = $eventFileItems.Count
        $finished = (Select-String -Path (Join-Path $clientLogDir "*_events.jsonl") `
            -Pattern '"event": "client_finished"' -ErrorAction SilentlyContinue |
            Measure-Object).Count
        $uploads = (Select-String -Path (Join-Path $clientLogDir "*_events.jsonl") `
            -Pattern '"event": "upload_done"' -ErrorAction SilentlyContinue |
            Measure-Object).Count
        $paramReads = (Select-String -Path (Join-Path $clientLogDir "*_events.jsonl") `
            -Pattern '"event": "param_read_response"' -ErrorAction SilentlyContinue |
            Measure-Object).Count
    }
}

Write-Output "clients=$clientTotal"
Write-Output "client_processes=$($rows.Count)"
Write-Output "running_client_processes=$running"
Write-Output "client_groups=$($groupRows.Count)"
Write-Output "running_client_groups=$runningGroups"
Write-Output "event_files=$eventFiles"
Write-Output "upload_done_events=$uploads"
Write-Output "param_read_events=$paramReads"
Write-Output "client_finished_events=$finished"
