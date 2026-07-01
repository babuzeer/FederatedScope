param(
    [string]$RepoDir = "",
    [string]$PythonBin = "python",
    [string]$RunId = "",
    [string]$CaseRoot = "",
    [string]$DataRoot = "D:\datasets\OfficeHomeDataset_10072016",
    [string]$ManifestRoot = "",
    [string]$ManifestMode = "manifest-only",
    [bool]$PrepareManifests = $true,
    [Parameter(Mandatory = $true)][string]$ServerHost,
    [string]$ServerBindHost = "0.0.0.0",
    [int]$ServerPort = 55051,
    [Parameter(Mandatory = $true)][string]$ClientHost,
    [string]$ClientHosts = "",
    [string]$ClientBindHost = "0.0.0.0",
    [string]$ClientBindHosts = "",
    [string]$ClientHostAssignment = "block",
    [int]$ClientPortBase = 56000,
    [int]$ClientBindPortBase = 0,
    [int]$ClientAdvertisePortBase = 0,
    [string]$ClientAdvertisePorts = "",
    [string]$ClientBindPorts = "",
    [int]$ClientNum = 60,
    [string]$Domains = "Art,Clipart,Product,Real_World",
    [int]$ClientsPerDomain = 15,
    [int]$TotalRounds = 100,
    [int]$SampleClients = 0,
    [int]$GenNum = 20,
    [int]$NumGeneratedPerSample = 0,
    [int]$NumGeneratedPerPrototype = 0,
    [int]$TargetSizePerClass = 0,
    [bool]$UseLds = $true,
    [double]$LdsAlpha = 0.1,
    [int]$Seed = 42,
    [string]$FeatureCacheDir = "exp/ggeur_headonly_real_cache/officehome_vitb16_60c_gen20_fcache_v1",
    [string]$HeadOnlyCacheVersion = "officehome_vitb16_60c_gen20_fcache_v1",
    [bool]$SkipRound0IfAugCache = $true,
    [string]$HeadOnlyEvalMode = "client",
    [int]$StageTimeout = 7200,
    [int]$JoinTimeoutSeconds = 900,
    [int]$ExtractBatchSize = 64,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 0,
    [int]$Device = 0,
    [bool]$UseGpu = $true
)

$ErrorActionPreference = "Stop"

if (-not $RepoDir) {
    $RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}
if (-not $RunId) {
    $RunId = "officehome_vit_headonly_60c_$(Get-Date -Format yyyyMMdd_HHmmss)"
}
if (-not $CaseRoot) {
    $CaseRoot = "scripts/distributed_scripts/ggeur_multimachine/runs/$RunId"
}
if (-not $ManifestRoot) {
    $ManifestRoot = "exp/headonly_system/manifests/$RunId"
}
if ($SampleClients -le 0) {
    $SampleClients = $ClientNum
}
if ($NumGeneratedPerSample -le 0) {
    $NumGeneratedPerSample = $GenNum
}
if ($NumGeneratedPerPrototype -le 0) {
    $NumGeneratedPerPrototype = $GenNum
}
if ($TargetSizePerClass -le 0) {
    $TargetSizePerClass = $GenNum
}
if ($ClientBindPortBase -le 0) {
    $ClientBindPortBase = $ClientPortBase
}
if ($ClientAdvertisePortBase -le 0) {
    $ClientAdvertisePortBase = $ClientPortBase
}
if (($ClientsPerDomain * 4) -ne $ClientNum) {
    throw "Expected 4 domains x ClientsPerDomain = ClientNum."
}

Set-Location $RepoDir
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CaseRoot) |
    Out-Null
New-Item -ItemType Directory -Force -Path $ManifestRoot | Out-Null

if ($PrepareManifests) {
    $manifestArgs = @(
        "scripts/prepare_officehome_client_manifests.py",
        "--source-root", $DataRoot,
        "--output-root", $ManifestRoot,
        "--client-num", "$ClientNum",
        "--splits", "0.7,0.0,0.3",
        "--seed", "$Seed",
        "--lds-alpha", "$LdsAlpha",
        "--lds-seed", "$Seed",
        "--domains", $Domains,
        "--mode", $ManifestMode
    )
    if ($UseLds) {
        $manifestArgs += "--use-lds"
    }
    & $PythonBin @manifestArgs
}

$genArgs = @(
    "scripts/distributed_scripts/ggeur_multimachine/generate_configs.py",
    "--output-root", $CaseRoot,
    "--run-id", $RunId,
    "--dataset", "officehome",
    "--model", "vit",
    "--method", "fedavg",
    "--head-only",
    "--server-host", $ServerHost,
    "--server-bind-host", $ServerBindHost,
    "--server-port", "$ServerPort",
    "--client-host", $ClientHost,
    "--client-hosts", $ClientHosts,
    "--client-host-assignment", $ClientHostAssignment,
    "--client-bind-host", $ClientBindHost,
    "--client-bind-hosts", $ClientBindHosts,
    "--client-port-base", "$ClientPortBase",
    "--client-bind-port-base", "$ClientBindPortBase",
    "--client-advertise-port-base", "$ClientAdvertisePortBase",
    "--client-advertise-ports", $ClientAdvertisePorts,
    "--client-bind-ports", $ClientBindPorts,
    "--clients", "$ClientNum",
    "--sample-clients", "$SampleClients",
    "--rounds", "$TotalRounds",
    "--stage-timeout", "$StageTimeout",
    "--join-timeout-seconds", "$JoinTimeoutSeconds",
    "--device", "$Device",
    "--seed", "$Seed",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--extract-batch-size", "$ExtractBatchSize",
    "--num-generated-per-sample", "$NumGeneratedPerSample",
    "--num-generated-per-prototype", "$NumGeneratedPerPrototype",
    "--target-size-per-class", "$TargetSizePerClass",
    "--lds-alpha", "$LdsAlpha",
    "--feature-cache-dir", $FeatureCacheDir,
    "--headonly-cache-version", $HeadOnlyCacheVersion,
    "--officehome-manifest-base", $ManifestRoot,
    "--officehome-domains", $Domains,
    "--headonly-eval-mode", $HeadOnlyEvalMode
)
if ($UseLds) {
    $genArgs += "--use-lds"
} else {
    $genArgs += "--no-use-lds"
}
if (-not $UseGpu) {
    $genArgs += "--no-use-gpu"
}
if (-not $SkipRound0IfAugCache) {
    $genArgs += "--no-headonly-skip-round0-if-augmented-cache-exists"
}

New-Item -ItemType Directory -Force -Path $CaseRoot | Out-Null
& $PythonBin @genArgs | Tee-Object -FilePath (Join-Path $CaseRoot "generate_output.json")

$info = @"
run_id=$RunId
case_root=$CaseRoot
data_root=$DataRoot
manifest_root=$ManifestRoot
client_num=$ClientNum
domains=$Domains
clients_per_domain=$ClientsPerDomain
total_rounds=$TotalRounds
gen_num=$GenNum
feature_cache_dir=$FeatureCacheDir
headonly_cache_version=$HeadOnlyCacheVersion
skip_round0_if_aug_cache=$SkipRound0IfAugCache
server_host=$ServerHost
server_bind_host=$ServerBindHost
server_port=$ServerPort
client_host=$ClientHost
client_hosts=$ClientHosts
client_bind_host=$ClientBindHost
client_port_base=$ClientPortBase
client_bind_port_base=$ClientBindPortBase
client_advertise_port_base=$ClientAdvertisePortBase
created_at=$(Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
"@
$info | Set-Content -Path (Join-Path $CaseRoot "final_headonly_60c_run_info.txt") -Encoding ascii

Write-Output "$CaseRoot/officehome_vit_fedavg_headonly"
