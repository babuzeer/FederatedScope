param(
    [switch]$Restart,
    [switch]$Foreground
)

$ErrorActionPreference = "Continue"

$RepoDir = "D:\Projects\FederatedScope"
$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe"
$ConfigPath = "scripts/headonly_cache_generation/officehome_vit_cachegen_400c_random25_gen20_8g.yaml"
$CmdPath = "D:\Projects\FederatedScope\scripts\headonly_cache_generation\run_400c_cachegen_8g.cmd"
$RunRoot = "D:\Projects\FederatedScope\exp\headonly_cache_generation\direct_400c_random25_20r_gen20_8g"
$LogDir = Join-Path $RunRoot "launcher_logs"
$StdoutLog = Join-Path $LogDir "stdout.log"
$StderrLog = Join-Path $LogDir "stderr.log"
$PidFile = Join-Path $LogDir "pid.txt"

Set-Location $RepoDir
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONPATH = $RepoDir
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:WANDB_MODE = "disabled"
$env:WANDB_DISABLED = "true"

$matchPattern = "*federatedscope.main*officehome_vit_cachegen_400c_random25_gen20_8g.yaml*"
$running = Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like $matchPattern } |
    Select-Object -ExpandProperty ProcessId

if ($running -and -not $Restart) {
    Write-Output "already_running=$($running -join ',')"
    Write-Output "pid_file=$PidFile"
    Write-Output "stdout=$StdoutLog"
    Write-Output "stderr=$StderrLog"
    exit 0
}

if ($running -and $Restart) {
    foreach ($procId in $running) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Output "stopped_existing=$procId"
    }
}

$ArgsList = @(
    "-m",
    "federatedscope.main",
    "--cfg",
    $ConfigPath
)

if ($Foreground) {
    & $PythonBin @ArgsList 2>&1 | Tee-Object -FilePath $StdoutLog
    exit $LASTEXITCODE
}

$cmdLine = "cmd.exe /d /c `"$CmdPath`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmdLine
    CurrentDirectory = $RepoDir
}

if ($result.ReturnValue -ne 0) {
    throw "Win32_Process.Create failed return_value=$($result.ReturnValue)"
}

$result.ProcessId | Set-Content -Path $PidFile
Write-Output "started_pid=$($result.ProcessId)"
Write-Output "pid_file=$PidFile"
Write-Output "stdout=$StdoutLog"
Write-Output "stderr=$StderrLog"
