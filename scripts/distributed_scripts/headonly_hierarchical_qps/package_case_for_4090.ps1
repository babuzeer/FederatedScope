param(
    [Parameter(Mandatory = $true)][string]$CaseDir,
    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
$CaseDir = (Resolve-Path $CaseDir).Path
$RunId = Split-Path $CaseDir -Leaf
if (-not $OutputZip) {
    $OutputZip = Join-Path $CaseDir ("server_bundle_{0}.zip" -f $RunId)
}

$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("headonly_hierarchical_qps_{0}" -f ([guid]::NewGuid().ToString("N")))
$TargetScriptDir = Join-Path $TempRoot "scripts\distributed_scripts\headonly_hierarchical_qps"
$TargetCaseDir = Join-Path $TargetScriptDir "runs\$RunId"
New-Item -ItemType Directory -Force -Path $TargetScriptDir | Out-Null
New-Item -ItemType Directory -Force -Path $TargetCaseDir | Out-Null

Copy-Item -Path (Join-Path $ScriptDir "headonly_hierarchical_qps.py") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "prepare_case.py") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "summarize_subserver_qps.py") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "launch_root_server.sh") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "launch_subserver.sh") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "start_4090.sh") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "status_case.sh") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "stop_case.sh") -Destination $TargetScriptDir
Copy-Item -Path (Join-Path $ScriptDir "README.md") -Destination $TargetScriptDir -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $CaseDir "configs") -Destination $TargetCaseDir -Recurse
Copy-Item -Path (Join-Path $CaseDir "manifest.json") -Destination $TargetCaseDir

if (Test-Path $OutputZip) {
    Remove-Item $OutputZip -Force
}
Compress-Archive -Path (Join-Path $TempRoot "*") -DestinationPath $OutputZip
Remove-Item $TempRoot -Recurse -Force
Write-Output "server_bundle=$OutputZip"
