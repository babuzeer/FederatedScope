param(
    [string]$SourceRoot = "D:\Projects\FederatedScope\exp\ggeur_headonly_real_cache\officehome_vitb16_400c_random25_gen20_fcache_v1",
    [string]$SaveRoot = "D:\Projects\FederatedScope\exp\saved_feature_caches\officehome_vitb16_400c_random25_gen20_fcache_v1_20260630_final",
    [string]$CacheVersion = "officehome_vitb16_400c_random25_gen20_fcache_v1"
)

$ErrorActionPreference = "Stop"

$SourceAugDir = Join-Path $SourceRoot "headonly_augmented\$CacheVersion"
$SaveAugDir = Join-Path $SaveRoot "headonly_augmented\$CacheVersion"
$ManifestPath = Join-Path $SaveRoot "cache_manifest_400c_saved.txt"
$ShaPath = Join-Path $SaveRoot "sha256_400c_augmented_pt.txt"

if (-not (Test-Path $SourceAugDir)) {
    throw "source augmented cache dir not found: $SourceAugDir"
}

$sourceFiles = Get-ChildItem $SourceAugDir -Filter "office-home_client_*.pt" -File | Sort-Object Name
if ($sourceFiles.Count -ne 400) {
    throw "expected 400 source cache files, got $($sourceFiles.Count): $SourceAugDir"
}

New-Item -ItemType Directory -Force -Path $SaveRoot | Out-Null
Copy-Item -Path (Join-Path $SourceRoot "*") -Destination $SaveRoot -Recurse -Force

$savedFiles = Get-ChildItem $SaveAugDir -Filter "office-home_client_*.pt" -File | Sort-Object Name
$ids = @()
foreach ($file in $savedFiles) {
    if ($file.Name -match "client_(\d+)\.pt$") {
        $ids += [int]$Matches[1]
    }
}

$missing = @()
foreach ($i in 1..400) {
    if ($ids -notcontains $i) {
        $missing += $i
    }
}

if ($savedFiles.Count -ne 400 -or $missing.Count -ne 0) {
    throw "saved cache incomplete: files=$($savedFiles.Count), missing=$($missing -join ',')"
}

$savedFiles | ForEach-Object {
    Set-ItemProperty -Path $_.FullName -Name IsReadOnly -Value $true
}

"source_root=$SourceRoot" | Set-Content $ManifestPath
"source_augmented_dir=$SourceAugDir" | Add-Content $ManifestPath
"saved_root=$SaveRoot" | Add-Content $ManifestPath
"saved_augmented_dir=$SaveAugDir" | Add-Content $ManifestPath
"file_count=$($savedFiles.Count)" | Add-Content $ManifestPath
"total_bytes=$(($savedFiles | Measure-Object Length -Sum).Sum)" | Add-Content $ManifestPath
"missing_count=$($missing.Count)" | Add-Content $ManifestPath
"missing_ids=$(($missing | ForEach-Object { $_.ToString() }) -join ',')" | Add-Content $ManifestPath
"saved_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" | Add-Content $ManifestPath
"files: name bytes mtime readonly" | Add-Content $ManifestPath
$savedFiles | ForEach-Object {
    "$($_.Name) $($_.Length) $($_.LastWriteTime.ToString('yyyy-MM-dd_HH:mm:ss')) $($_.IsReadOnly)"
} | Add-Content $ManifestPath

$savedFiles | ForEach-Object {
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
    "$($hash.Hash)  $($_.Name)"
} | Set-Content $ShaPath

Write-Output "saved_root=$SaveRoot"
Write-Output "saved_augmented_dir=$SaveAugDir"
Write-Output "file_count=$($savedFiles.Count)"
Write-Output "total_bytes=$(($savedFiles | Measure-Object Length -Sum).Sum)"
Write-Output "missing_count=$($missing.Count)"
Write-Output "manifest=$ManifestPath"
Write-Output "sha256=$ShaPath"
