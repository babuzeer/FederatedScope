param(
    [string]$SourceRoot = "D:\Projects\FederatedScope",
    [string]$OutputDir = "D:\Projects\FederatedScope\dist",
    [string]$BundleName = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SourceRoot)) {
    throw "SourceRoot does not exist: $SourceRoot"
}

$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

if (-not $BundleName) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BundleName = "FederatedScope_headonly_two_machine_$stamp.zip"
}

$OutputZip = Join-Path $OutputDir $BundleName
$StageRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "FederatedScope_bundle_" + [System.Guid]::NewGuid().ToString("N"))

$includeDirs = @(
    "benchmark",
    "doc",
    "docs",
    "environment",
    "federatedscope",
    "ggeur_standalone",
    "scripts",
    "tests"
)

$includeRootPatterns = @(
    "*.py",
    "*.md",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.ini",
    "*.txt",
    "LICENSE",
    "meta.yaml",
    "setup.py",
    ".flake8",
    ".pre-commit-config.yaml",
    ".style.yapf"
)

$excludeDirNames = @(
    ".git",
    ".idea",
    ".vscode",
    ".claude",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "exp",
    "data",
    "OfficeHomeDataset_10072016",
    "tmp_download_bench"
)

$excludeFilePatterns = @(
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.pid",
    "*.pt",
    "*.pth",
    "*.npz",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.7z",
    "*.ipynb_checkpoints"
)

function Test-ExcludedPath {
    param([string]$Path)

    $parts = $Path -split '[\\/]+'
    foreach ($part in $parts) {
        if ($excludeDirNames -contains $part) {
            return $true
        }
    }

    $leaf = Split-Path -Leaf $Path
    foreach ($pattern in $excludeFilePatterns) {
        if ($leaf -like $pattern) {
            return $true
        }
    }
    return $false
}

function Copy-TreeFiltered {
    param(
        [string]$From,
        [string]$To
    )

    if (Test-ExcludedPath $From) {
        return
    }

    $item = Get-Item -LiteralPath $From -Force
    if ($item.PSIsContainer) {
        New-Item -ItemType Directory -Force -Path $To | Out-Null
        Get-ChildItem -LiteralPath $From -Force | ForEach-Object {
            Copy-TreeFiltered -From $_.FullName -To (Join-Path $To $_.Name)
        }
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $To) |
            Out-Null
        Copy-Item -LiteralPath $From -Destination $To -Force
    }
}

if (Test-Path -LiteralPath $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

try {
    foreach ($dir in $includeDirs) {
        $src = Join-Path $SourceRoot $dir
        if (Test-Path -LiteralPath $src) {
            Copy-TreeFiltered -From $src -To (Join-Path $StageRoot $dir)
        }
    }

    foreach ($pattern in $includeRootPatterns) {
        Get-ChildItem -LiteralPath $SourceRoot -File -Force -Filter $pattern |
            ForEach-Object {
                if (-not (Test-ExcludedPath $_.FullName)) {
                    Copy-Item -LiteralPath $_.FullName `
                        -Destination (Join-Path $StageRoot $_.Name) -Force
                }
            }
    }

    if (Test-Path -LiteralPath $OutputZip) {
        Remove-Item -LiteralPath $OutputZip -Force
    }
    Compress-Archive -Path (Join-Path $StageRoot '*') `
        -DestinationPath $OutputZip -Force

    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $OutputZip
    $shaPath = "$OutputZip.sha256"
    "$($hash.Hash)  $(Split-Path -Leaf $OutputZip)" |
        Set-Content -LiteralPath $shaPath -Encoding ascii

    Write-Host "Wrote $OutputZip"
    Write-Host "SHA256 $($hash.Hash)"
    Write-Host "Wrote $shaPath"
} finally {
    if (Test-Path -LiteralPath $StageRoot) {
        Remove-Item -LiteralPath $StageRoot -Recurse -Force
    }
}
