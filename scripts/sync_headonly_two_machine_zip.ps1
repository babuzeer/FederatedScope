param(
    [ValidateSet("Both", "Server", "Client")]
    [string]$Target = "Both",
    [string]$SourceRoot = "D:\Projects\FederatedScope",
    [string]$BundleName = "FederatedScope_headonly_two_machine_latest.zip",
    [string]$ServerLogin = "root@10.112.81.135",
    [string]$ServerUploadDir = "/root/autodl-tmp",
    [string]$ServerProjectDir = "/root/autodl-tmp/FederatedScope",
    [string]$ClientLogin = "fsuser@10.129.222.189",
    [string]$ClientProjectDir = "D:\Projects\FederatedScope",
    [switch]$BackupExistingProject,
    [switch]$OverlayOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$distDir = Join-Path $SourceRoot "dist"
$zipPath = Join-Path $distDir $BundleName

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Body
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    if ($DryRun) {
        return
    }
    & $Body
}

Invoke-Step "Package source zip" {
    & powershell -NoProfile -ExecutionPolicy Bypass `
        -File (Join-Path $SourceRoot "scripts\package_headonly_two_machine_bundle.ps1") `
        -SourceRoot $SourceRoot `
        -OutputDir $distDir `
        -BundleName $BundleName
}

if ($DryRun) {
    Write-Host "DryRun zip path: $zipPath"
}

if ($Target -eq "Both" -or $Target -eq "Server") {
    $serverZip = "$ServerUploadDir/$BundleName"
    $backupServer = if ($BackupExistingProject) { "1" } else { "0" }
    $overlayServer = if ($OverlayOnly) { "1" } else { "0" }
    $serverCommand = @"
set -e
if [ "$backupServer" = "1" ] && [ -d "$ServerProjectDir" ]; then
  mv "$ServerProjectDir" "${ServerProjectDir}_old_`$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$ServerProjectDir"
if [ "$overlayServer" != "1" ]; then
  for d in benchmark doc docs environment federatedscope ggeur_standalone scripts tests; do
    rm -rf "$ServerProjectDir/`$d"
  done
  find "$ServerProjectDir" -maxdepth 1 -type f \( \
    -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' \
    -o -name '*.toml' -o -name '*.ini' -o -name '*.txt' \
    -o -name 'LICENSE' -o -name 'meta.yaml' -o -name 'setup.py' \
    -o -name '.flake8' -o -name '.pre-commit-config.yaml' \
    -o -name '.style.yapf' \) -delete
fi
unzip -oq "$serverZip" -d "$ServerProjectDir"
echo "deployed_to=$ServerProjectDir"
"@

    Invoke-Step "Upload zip to server $ServerLogin" {
        & scp $zipPath "${ServerLogin}:$serverZip"
    }
    Invoke-Step "Deploy zip on server $ServerLogin" {
        & ssh $ServerLogin $serverCommand
    }
}

if ($Target -eq "Both" -or $Target -eq "Client") {
    $remoteHomeZip = $BundleName
    $backupClient = if ($BackupExistingProject) { '$true' } else { '$false' }
    $overlayClient = if ($OverlayOnly) { '$true' } else { '$false' }
    $clientCommand = @"
`$ErrorActionPreference = 'Stop'
`$project = '$ClientProjectDir'
`$zip = Join-Path `$env:USERPROFILE '$remoteHomeZip'
if ($backupClient -and (Test-Path -LiteralPath `$project)) {
    `$parent = Split-Path -Parent `$project
    `$name = Split-Path -Leaf `$project
    `$backup = Join-Path `$parent (`$name + '_old_' + (Get-Date -Format yyyyMMdd_HHmmss))
    Rename-Item -LiteralPath `$project -NewName (Split-Path -Leaf `$backup)
}
New-Item -ItemType Directory -Force -Path `$project | Out-Null
if (-not $overlayClient) {
    `$codeDirs = @(
        'benchmark', 'doc', 'docs', 'environment', 'federatedscope',
        'ggeur_standalone', 'scripts', 'tests'
    )
    foreach (`$dir in `$codeDirs) {
        `$path = Join-Path `$project `$dir
        if (Test-Path -LiteralPath `$path) {
            Remove-Item -LiteralPath `$path -Recurse -Force
        }
    }
    `$patterns = @(
        '*.py', '*.md', '*.yaml', '*.yml', '*.toml', '*.ini', '*.txt',
        'LICENSE', 'meta.yaml', 'setup.py', '.flake8',
        '.pre-commit-config.yaml', '.style.yapf'
    )
    Get-ChildItem -LiteralPath `$project -File -Force | Where-Object {
        `$name = `$_.Name
        `$matched = `$false
        foreach (`$pattern in `$patterns) {
            if (`$name -like `$pattern) {
                `$matched = `$true
                break
            }
        }
        `$matched
    } | Remove-Item -Force
}
Expand-Archive -LiteralPath `$zip -DestinationPath `$project -Force
Write-Output "deployed_to=`$project"
"@

    Invoke-Step "Upload zip to client $ClientLogin" {
        & scp $zipPath "${ClientLogin}:$remoteHomeZip"
    }
    Invoke-Step "Deploy zip on client $ClientLogin" {
        & ssh $ClientLogin powershell -NoProfile -ExecutionPolicy Bypass -Command $clientCommand
    }
}
