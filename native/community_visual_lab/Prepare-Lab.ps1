param(
    [Parameter(Mandatory)]
    [string] $SdkProject,
    [Parameter(Mandatory)]
    [string] $Lab
)

$ErrorActionPreference = 'Stop'
$SdkProject = (Resolve-Path -LiteralPath $SdkProject).Path
$Lab = [IO.Path]::GetFullPath($Lab)
if (Test-Path -LiteralPath $Lab) {
    throw "Choose a new directory for the isolated SDK lab: $Lab"
}
foreach ($name in 'include', 'lib', 'bin', 'bin/D3D12', 'bin/plugins', 'source', 'data/core.ung', 'data/PCG.world') {
    if (!(Test-Path -LiteralPath (Join-Path $SdkProject $name))) {
        throw "Required SDK project path is missing: $name"
    }
}

foreach ($name in 'stock/source', 'stock/bin/screenshots', 'data/configs', 'logs') {
    New-Item -ItemType Directory -Path (Join-Path $Lab $name) -Force | Out-Null
}
foreach ($name in 'include', 'lib') {
    New-Item -ItemType Junction -Path "$Lab/stock/$name" -Target "$SdkProject/$name" | Out-Null
}
foreach ($name in 'D3D12', 'plugins') {
    New-Item -ItemType Junction -Path "$Lab/stock/bin/$name" -Target "$SdkProject/bin/$name" | Out-Null
}
Get-ChildItem -LiteralPath "$SdkProject/bin" -File -Filter '*.dll' |
    Where-Object Name -NotMatch '^(EditorCore|Qt6|Varjo)' |
    Copy-Item -Destination "$Lab/stock/bin"
$templateFiles = "$SdkProject/source/*.cpp", "$SdkProject/source/*.h", "$SdkProject/source/CMakeLists.txt"
Copy-Item -Path $templateFiles -Destination "$Lab/stock/source"
Copy-Item -LiteralPath "$SdkProject/source/cmake" -Destination "$Lab/stock/source" -Recurse
foreach ($name in 'core.ung', 'scripts.ung', 'root_mount.umount') {
    Copy-Item -LiteralPath "$SdkProject/data/$name" -Destination "$Lab/data"
}
Get-ChildItem -LiteralPath "$SdkProject/data/configs" -File |
    Where-Object Extension -Ne '.meta' |
    Copy-Item -Destination "$Lab/data/configs"
[xml] $existingWorld = Get-Content -LiteralPath "$SdkProject/data/PCG.world" -Raw
$render = $existingWorld.SelectSingleNode('/world/render')
if (!$render) { throw 'The source world has no renderer profile' }
$flatWorld = @"
<?xml version="1.0" encoding="utf-8"?>
<world version="2.22.0.0">
    $($render.OuterXml)
    <editor/>
</world>
"@
[IO.File]::WriteAllText("$Lab/data/PCG_Flat.world", $flatWorld)

Write-Output "Prepared isolated SDK lab: $Lab"
