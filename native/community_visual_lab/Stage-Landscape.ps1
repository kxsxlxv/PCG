param(
    [Parameter(Mandatory)]
    [string] $SdkProject,
    [Parameter(Mandatory)]
    [string] $Lab
)

$ErrorActionPreference = 'Stop'
$SdkProject = (Resolve-Path -LiteralPath $SdkProject).Path
$Lab = (Resolve-Path -LiteralPath $Lab).Path
$copy = Join-Path $Lab 'data_landscape'
if (Test-Path -LiteralPath $copy) {
    throw "Landscape staging already exists at $copy"
}
New-Item -ItemType Directory -Path $copy | Out-Null
$resources = @(
    'PCG.world', 'PCG.world.meta', 'guids.db', 'core.ung', 'scripts.ung', 'root_mount.umount',
    'configs', 'template_assets', 'art_samples', 'showcase_content', '.runtimes', 'ComponentSystem'
)
foreach ($resource in $resources) {
    $from = Join-Path "$SdkProject/data" $resource
    Copy-Item -LiteralPath $from -Destination $copy -Recurse -Force
}
$original = (Get-FileHash -LiteralPath "$SdkProject/data/PCG.world").Hash
$staged = (Get-FileHash -LiteralPath "$copy/PCG.world").Hash
if ($original -ne $staged) {
    throw 'Landscape world changed while it was being copied'
}
Write-Output "Staged existing LandscapeTerrain and its project resources. PCG.world SHA-256 $staged"
