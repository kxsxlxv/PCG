param(
    [Parameter(Mandatory)]
    [string] $Lab,
    [ValidateSet('native', 'marker', 'day', 'night', 'terrain-probe')]
    [string] $Mode = 'day',
    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Release',
    [ValidateSet('dx12', 'vk')]
    [string] $Renderer = 'dx12',
    [ValidateSet('flat_preview', 'landscape_local')]
    [string] $Placement = 'flat_preview',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string] $Package = 'preview-v2',
    [double] $OriginX = 0,
    [double] $OriginY = 0,
    [switch] $Validate,
    [switch] $SaveScene,
    [switch] $OfficialImporter,
    [switch] $Capture
)

$ErrorActionPreference = 'Stop'
$Lab = (Resolve-Path -LiteralPath $Lab).Path
$runtime = Join-Path $Lab 'stock/bin'
$exe = if ($Configuration -eq 'Debug') { 'PCGVisualLabd.exe' } else { 'PCGVisualLab.exe' }
$tag = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$log = Join-Path $Lab "logs/$Mode-$tag.log"
$capturePath = if ($Capture) { Join-Path $runtime "screenshots/$Mode-$tag" } else { '' }
if ($SaveScene -and $Mode -notin @('day', 'night')) { throw 'SaveScene requires a building preview' }
if ($OfficialImporter -and $Placement -eq 'landscape_local') {
    throw 'The official importer diagnostic runs on the flat reference'
}
$save = if ($SaveScene) { "PCG_Preview_$Mode-$tag" } else { '' }
$data = 'data'
$world = 'PCG_Flat'
if ($Mode -eq 'terrain-probe' -or $Placement -eq 'landscape_local') {
    if ($Capture -and $Mode -eq 'terrain-probe') { throw 'Terrain probe emits numerical diagnostics' }
    $data = 'data_landscape'
    $world = 'PCG'
}
if (!(Test-Path -LiteralPath "$Lab/$data/$world.world" -PathType Leaf)) {
    throw "Missing staged world: $Lab/$data/$world.world"
}
$usesPackage = $Mode -ne 'native' -and $Mode -ne 'terrain-probe'
if ($usesPackage -and !(Test-Path -LiteralPath "$Lab/data/$Package/expected.txt")) {
    throw "Missing generated preview package: $Lab/data/$Package"
}
if ($usesPackage -and $data -eq 'data_landscape' -and !(Test-Path -LiteralPath "$Lab/$data/$Package")) {
    New-Item -ItemType Junction -Path "$Lab/$data/$Package" -Target "$Lab/data/$Package" | Out-Null
}
$culture = [Globalization.CultureInfo]::InvariantCulture
$environment = @{
    PCG_LAB_MODE = $Mode
    PCG_LAB_CAPTURE = $capturePath
    PCG_LAB_PLACEMENT = $Placement
    PCG_LAB_PACKAGE = $Package
    PCG_LAB_ORIGIN_X = $OriginX.ToString($culture)
    PCG_LAB_ORIGIN_Y = $OriginY.ToString($culture)
    PCG_LAB_VALIDATE = ([int] ($Validate -or $SaveScene)).ToString()
    PCG_LAB_SAVE_SCENE = $save
    PCG_LAB_OFFICIAL_IMPORTER = ([int] $OfficialImporter.IsPresent).ToString()
}
$previous = @{}
$wait = $Capture -or $Validate -or $SaveScene -or $Mode -eq 'terrain-probe'

try {
    foreach ($key in $environment.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key)
        [Environment]::SetEnvironmentVariable($key, $environment[$key])
    }
    $arguments = @(
        "-video_app $Renderer", '-video_offscreen 0', '-main_window_fullscreen 0',
        '-main_window_size 1280 720', '-sound_app null', '-extern_plugin UnigineGLTFImporter',
        "-data_path `"$Lab/$data/`"", "-engine_log `"$log`"",
        "-console_command `"config_autosave 0 && background_update 2 && splash_screen 0 && world_load $world`""
    )
    $launch = @{
        FilePath = Join-Path $runtime $exe
        WorkingDirectory = $runtime
        ArgumentList = $arguments
        PassThru = $true
    }
    if (!$Capture -and $wait) { $launch.WindowStyle = 'Hidden' }
    $process = Start-Process @launch
    Write-Output "Engine log: $log"
    if ($wait) {
        $process.WaitForExit()
        Get-Content -LiteralPath $log -Tail 18
        if ($process.ExitCode -ne 0) { exit $process.ExitCode }
    }
    if ($Capture) {
        if (!(Test-Path -LiteralPath $capturePath) -or
            !(Get-ChildItem -LiteralPath $capturePath -Recurse -Filter '*.png' -File)) {
            throw "The engine did not save a PNG capture. See $log"
        }
        Write-Output "Captures: $capturePath"
    }
    if ($SaveScene) {
        $worldPath = "$Lab/$data/$save.world"
        $nodePath = "$Lab/$data/$save.node"
        [xml] $worldXml = Get-Content -LiteralPath $worldPath -Raw
        [xml] $nodeXml = Get-Content -LiteralPath $nodePath -Raw
        $root = $nodeXml.SelectSingleNode('/*[self::node] | /*/node')
        if (!$root -or $root.GetAttribute('name') -ne 'PCGPreview') {
            throw "Missing generated preview root in $nodePath"
        }
        $editor = $worldXml.SelectSingleNode('/world/editor')
        if (!$editor) { $editor = $worldXml.DocumentElement.AppendChild($worldXml.CreateElement('editor')) }
        if (!$editor.SelectSingleNode('node[@name="PCGPreview"]')) {
            $editor.AppendChild($worldXml.ImportNode($root, $true)) | Out-Null
        }
        $worldXml.Save($worldPath)
        Write-Output "Saved Editor world: $worldPath"
    }
}
finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key])
    }
}
