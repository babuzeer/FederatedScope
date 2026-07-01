param(
    [Parameter(Mandatory = $true)][string]$CaseDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$CaseDir = (Resolve-Path $CaseDir).Path
$PidFile = Join-Path $CaseDir "system\client_pids.csv"
$GroupPidFile = Join-Path $CaseDir "system\client_group_pids.csv"
$StoppedCount = 0

function Stop-PidRows {
    param(
        [string]$Path,
        [string]$Kind
    )
    if (-not (Test-Path $Path)) {
        Write-Output "$($Kind)_pid_file_missing=$Path"
        return
    }
    $rows = Import-Csv $Path
    foreach ($row in $rows) {
        $pidValue = [int]$row.pid
        if ($pidValue -le 0) {
            continue
        }
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -eq $proc) {
            if ($Kind -eq "client_group") {
                Write-Output "already stopped client_group=$($row.group_id) pid=$pidValue"
            } else {
                Write-Output "already stopped client=$($row.client_id) pid=$pidValue"
            }
            continue
        }
        Stop-Process -Id $pidValue -Force:$Force -ErrorAction SilentlyContinue
        $script:StoppedCount += 1
        if ($Kind -eq "client_group") {
            Write-Output "stopped client_group=$($row.group_id) pid=$pidValue"
        } else {
            Write-Output "stopped client=$($row.client_id) pid=$pidValue"
        }
    }
}

Stop-PidRows -Path $PidFile -Kind "client"
Stop-PidRows -Path $GroupPidFile -Kind "client_group"
if ($StoppedCount -eq 0) {
    Write-Output "nothing_to_stop=1"
}
