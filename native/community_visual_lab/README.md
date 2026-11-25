# UNIGINE Community visual lab

This lab runs the existing Python compiler offline, converts its verified GLB geometry into native UNIGINE surfaces, and previews generated buildings on a flat reference or the existing LandscapeTerrain copy. Building geometry stays local in meters and Z-up. Placement uses double-precision transforms. The installed Community SDK was confirmed for this use. Sim integration and the real Adler data gate remain separate.

## Build and regenerate

Run from the PCG-core checkout in PowerShell 7. The commands use the installed Windows SDK template and the pinned Lab Python dependencies. The workstation already has a prepared lab under the SDK project's `pcg-local` directory.

```powershell
$SdkProject = (Resolve-Path -LiteralPath '../PCG').Path
$Lab = Join-Path $SdkProject 'pcg-local'
$Native = './native/community_visual_lab'
if (!(Test-Path -LiteralPath $Lab)) {
    & "$Native/Prepare-Lab.ps1" -SdkProject $SdkProject -Lab $Lab
}
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r lab/requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
cmake -S "$Lab/stock/source" -B "$Lab/stock/build" -G 'Visual Studio 18 2026' -A x64
cmake --build "$Lab/stock/build" --config Debug
cmake --build "$Lab/stock/build" --config Release
cmake -S $Native -B "$Lab/build" -G 'Visual Studio 18 2026' -A x64 "-DPCG_SDK_PROJECT=$SdkProject" "-DPCG_RUNTIME_DIR=$Lab/stock/bin"
cmake --build "$Lab/build" --config Release
cmake --build "$Lab/build" --config Debug
& .\.venv\Scripts\python.exe "$Native/export_preview.py" "$Lab/data/preview-v2" --fixture neighborhood
& "$Native/Run-Preview.ps1" -Lab $Lab -Mode day -Capture -SaveScene
```

Reuse the existing `.venv` and existing packages where present. Generate into a fresh package directory for each changed fixture, seed or comparison, then pass its directory name to `-Package`. The seed controls occupancy. Use `--fixture profiles` for one building of each available profile, `--profile house` for the house alone, `--lod 1` for the coarser authored detail, or `--input-fixture` for a local edited copy of the L4 input. The runtime accepts `-Mode day` or `-Mode night` and a shared `-OriginX` and `-OriginY`. Interactive F1–F4 select aerial, street, facade and roof. F5 clears and rebuilds the package's node tree. F6 selects terrain contact and diagnostics. Escape closes the preview. `-Validate` runs a rebuild and checks the resulting node count. `-SaveScene` writes a new Editor world, its preview node tree and referenced native meshes and materials under the staged data root.

```powershell
& .\.venv\Scripts\python.exe "$Native/export_preview.py" "$Lab/data/house-seed29-lod1" --profile house --seed 29 --lod 1
& "$Native/Run-Preview.ps1" -Lab $Lab -Package house-seed29-lod1 -Mode night
if (!(Test-Path -LiteralPath "$Lab/data_landscape")) {
    & "$Native/Stage-Landscape.ps1" -SdkProject $SdkProject -Lab $Lab
}
& "$Native/Run-Preview.ps1" -Lab $Lab -Mode terrain-probe
& "$Native/Run-Preview.ps1" -Lab $Lab -Mode day -Placement landscape_local -OriginX 680 -OriginY 680 -Capture -SaveScene
```

For L5, pass the local compiler's output directory containing `report.json`. The adapter checks each GLB's exact SHA-256, triangle count and building-local bounds. It applies the documented Y-up to Z-up mapping, preserves the original `WorldFrame` and canonical anchors in `gis_preview.json`, and subtracts one shared XY preview origin. The selected preview placement mode supplies base Z.

```powershell
& .\.venv\Scripts\python.exe "$Native/import_gis_preview.py" artifacts/l5-pilot/compiled "$Lab/data/preview-l56"
& "$Native/Run-Preview.ps1" -Lab $Lab -Mode day -Package preview-l56 -Capture -SaveScene
```

## Measured evidence

2026-11-25. Core baseline `d1d09387e735d71cde46d818d6a8babaf024fdca`. SDK Browser package Community 2.22.0.2, internal version 59, CMake C++ x64 double precision. The local files report engine runtime 2.22.0.0. Windows 10 build 19045, Core i7-13700KF, 32 GiB RAM, RTX 4070 Ti SUPER driver 621.24, Direct3D 12 at 1280 × 720. CMake 4.2.3, Visual Studio 18.10, MSVC 19.51.36260, Windows SDK 10.0.26100.0, CPython 3.11.17, uv 0.12.22, Node 24.13.0. Exact versions and logs are local.

| Gate | State | Evidence |
|---|---|---|
| Offline baseline | PASS | 49 unit tests, 7 research probes and the synthetic local compiler ran. Windows autocrlf had altered the pinned GeoJSON bytes. The repository now checks out the fixtures with LF and verifies the unchanged hashes. |
| G0 app | PASS | Stock Debug and Release CMake builds exited 0. Both loaded `PCG_Flat.world` on D3D12 and exited 0. Logs `stock-configure.log`, `stock-build-*.log`, `stock-verified-*.log`. |
| G1 native MESH | PASS | Asymmetric roof and wall round-trip preserved 2 triangles, 2 surface roles, 4 × 2.5 × 1.25 m bounds, indices, UV0 and hard normals reconstructed from tangent quaternions. The DX12 Debug capture is `native-20261125-154059-056/fixture.png`. |
| G2 official standalone import | FAIL | The plugin loaded in Debug and Release. `Import::isSupportedExtension("glb")` returned true and `createImporterByFileName` returned null for the asymmetric file. The run with the documented absolute input form is `marker-20261125-151949-053.log`. The Editor import workflow was outside the standalone test. |
| G2 direct mesh fallback | PASS for geometry and baseline render | The asymmetric GLB round-trip kept bounds (1, 2, 3)–(5, 4.5, 4.25) m and both groups. The L4 six-building day package retained 32,572 triangles and 11 shared material roles. Aerial, street, facade and roof captures show hip roofs, apertures, balconies and the courtyard. The L5.6 house and office used 2,524 and 7,252 triangles. Full channel, mip and color-space validation remains open. |
| Local Cartesian terrain | PASS for sampled placement | Five near-origin controls agreed within the predeclared 0.15 m tolerance. Worst absolute method difference was 0.000043 m. Both methods returned no height at (100000, 100000). Six buildings passed the 0.12 m sampled gap limit at preview origin (680, 680). Terrain is the Community world copy. |
| G3–G6 | NOT_RUN | Shared clusters, per-instance shader parameters, categorical attributes and arrays/mips still need their grouped native measurements. |
| G7–G12 and real L5 | NOT_RUN / BLOCKED | Full synthetic terrain-grid precision and Sim host tests are unrun. Real L5 remains blocked on authorized source data, matching terrain and independent reference evidence. [Issue #20](https://github.com/kxsxlxv/PCG/issues/20) |

For G0 before the flat preview starting world was corrected, D3D12 required the SDK's adjacent `D3D12` directory. The minimal world initialized rendering with white captures. Using the copied world's renderer profile without its source nodes produced usable captures. The frame evidence comes from UNIGINE's viewport render target and GPU image transfer. Earlier white captures remain in the local raw log.

## Terrain and visual review

`PCG.world` has one enabled `ObjectLandscapeTerrain` named `LandscapeTerrain`. The terrain node has an identity transform. Its ground, airfield ground and rotated airfield layer maps are staged with the world. The original scene and staged source world SHA-256 both equal `51FB4FF3E22BE2ECF110EEFC70E1BF60754B7E0724B1B19A34CE55E2988A85DE`. The staged loader reports stale runtime GUID entries. The landscape height and visible terrain resources loaded for the measured patch.

The provider transforms a vertical world-space ray into terrain-object local coordinates, queries only surface 0 with `Object::getIntersection`, then transforms the hit back to world Z. `LandscapeFetch::fetchForce` is the comparison path. Both use the same terrain source. The fixture checks the sampled horizontal location to 2 cm. The top-level transform and the actual layer-map transforms are exercised only as installed. No separate rotated terrain-object fixture was used. At most 2 m between densified boundary vertices, plus a representative point and a bounded interior lattice. Placement uses the maximum sampled Z with vertical walls and logs each point, coverage and relief. The footprint samples bound observed gaps. They do not establish continuous clearance or foundation adequacy.

| Building | Samples | Minimum Z (m) | Base Z (m) | Sampled relief (m) |
|---|---:|---:|---:|---:|
| block-a | 53 | 1300.15570 | 1300.20374 | 0.04804 |
| block-b | 59 | 1300.12094 | 1300.16578 | 0.04484 |
| block-c | 69 | 1300.09583 | 1300.13048 | 0.03465 |
| block-d | 61 | 1300.11529 | 1300.15654 | 0.04125 |
| block-e | 93 | 1300.08338 | 1300.12458 | 0.04120 |
| block-f | 66 | 1300.07292 | 1300.09782 | 0.02490 |

`day-20261125-153458-079.log` records 401 samples and a 6-to-6 rebuild. At origin (0, 0), the same 53-point house had 1.41119 m of sampled relief and was rejected. At (100000, 100000), all 53 samples lacked coverage and no node was placed. Both rejection cases remained reproducible after a rebuild. See `day-20261125-153518-918.log` and `day-20261125-153522-761.log`. The night house shows distinct warm and neutral emissive panes, with the same horizontal base. Some close views show hard shadow edges and repeat patterns. The selected patch is on the sample airfield and serves as a placement control.

`determinism.log` records six byte-identical files from separate seed-17 builds. At the same LOD, seed 29 preserves daytime geometry and changes emitted window assignments. The seed-29 LOD1 package also ran a 1-to-1 rebuild on LandscapeTerrain.

## Library comparison

```powershell
& .\.venv\Scripts\python.exe "$Native/catalog_assets.py" $SdkProject "$Lab/asset-shortlist-new.json"
& .\.venv\Scripts\python.exe "$Native/export_preview.py" "$Lab/data/house-plaster-comparison" --profile house --plaster-pack "$SdkProject/materials/PaintedPlaster015_2K-PNG.zip"
& "$Native/Run-Preview.ps1" -Lab $Lab -Package house-plaster-comparison -Mode day -Capture
```

The shortlist records 13 inspected roots and 25 selected files including project GLBs, props, HDRI, D: material selections and photos. The locally owned archives and source models remain in their original directories. `third-party-inventory.json` records 15 existing source checkouts. CGAL's checked-out revision is `22d5a1d375bcec3dc8d852f89ee70e55444c60a9`, which differs from `REVISIONS.txt`. This increment uses the existing PCG compiler and local SDK. No newly cloned geometry dependency enters the compiler.

The same house geometry, cameras, lighting, exposure and provisional native repeat were used for procedural cream plaster and ambientCG Painted Plaster 015. The asset has visibly sharper, higher-contrast deterioration that reads poorly as the default for an entire facade. The procedural recipe stays the default. The sidecar records the exact archive SHA and repeat. The adapter flips the green channel for normal maps, remaps ORM into the documented mesh_base metalness, roughness and specular channels, and binds AO with the installed SDK's `ao_map` state. Raw PNG material paths yielded a non-sRGB GPU texture in the runtime. The standalone adapter explicitly loads albedo with `FORMAT_SRGB` and checks the retained flag. The saved Editor material uses the serialized texture paths. Editor color parity and the asymmetric GPU channel and mip tests remain open. [Painted Plaster 015](https://ambientcg.com/view?id=PaintedPlaster015) · [ambientCG license](https://docs.ambientcg.com/license/)

The inspected Adler residence photo R001 has layered red roofing, exterior stair and balcony runs, wall-mounted HVAC and more pronounced eaves than the baseline. It is qualitative reference evidence. Leaking009's opacity image has sparse vertical streaks growing out of a horizontal band. It is a candidate for localized weathering after orientation review. That mask and the project HDRI were inventoried without being applied.

The tevy apartment, wall-bracket AC and roof vents declare CC0 in their GLB metadata. Blender 5.3 decoded and round-tripped 53,690, 2,922 and 454 triangles respectively. Some packed KTX2 images, including the sampled ORM textures, could not be decoded by that run. The separate native reference gallery therefore uses a uniform diagnostic material and keeps original scale. It establishes geometry and fit reference only. [tevy license](https://tevy.dev/)

```powershell
$Blender = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
& $Blender --background --factory-startup --python-exit-code 1 --python "$Native/decode_reference_glbs.py" -- "$Lab/decoded" "$SdkProject/buildings/apartment-block-v2-lod0.glb" "$SdkProject/props/ac-unit-on-wall-bracket-v1-lod0.glb" "$SdkProject/props/rooftop-vent-stacks-v1-lod0.glb"
& .\.venv\Scripts\python.exe "$Native/export_reference_preview.py" "$Lab/data/gallery-building" "$Lab/decoded/apartment-block-v2-lod0.glb"
& .\.venv\Scripts\python.exe "$Native/export_reference_preview.py" "$Lab/data/gallery-props" "$Lab/decoded/ac-unit-on-wall-bracket-v1-lod0.glb" "$Lab/decoded/rooftop-vent-stacks-v1-lod0.glb"
& "$Native/Run-Preview.ps1" -Lab $Lab -Package gallery-building -Mode day -Capture
& "$Native/Run-Preview.ps1" -Lab $Lab -Package gallery-props -Mode day -Capture
```

## Local outputs

All raw engine logs live under `pcg-local/logs`. Full-resolution captures live under `pcg-local/stock/bin/screenshots`. The SDK project and generated resources stay outside Git.
The stock Release runtime reopened both saved worlds and exited 0. See `saved-reopen-flat.log` and `saved-reopen-terrain.log`.

| Artifact | Local directory or file below pcg-local |
|---|---|
| Editor world, six flat buildings | `data/PCG_Preview_day-20261125-155337-490.world` |
| Editor world, six terrain buildings | `data_landscape/PCG_Preview_day-20261125-155340-476.world` |
| Flat day captures | `stock/bin/screenshots/day-20261125-152912-142` |
| Terrain house day and contact | `stock/bin/screenshots/day-20261125-153526-547` |
| Terrain house night | `stock/bin/screenshots/night-20261125-155404-076` |
| Measured terrain neighborhood and green sample overlay | `stock/bin/screenshots/day-20261125-155340-476` |
| Earlier terrain neighborhood with raw albedo path | `stock/bin/screenshots/day-20261125-154704-444` |
| Rejected slope and red sample overlay | `stock/bin/screenshots/day-20261125-155045-659/contact.png` |
| L5.6 synthetic captures | `stock/bin/screenshots/day-20261125-154207-258` |
| Plaster A/B facade captures | `stock/bin/screenshots/day-20261125-154226-034` and `day-20261125-154244-424` |
| Native donor gallery | `stock/bin/screenshots/day-20261125-154623-343` and `day-20261125-154642-548` |
| Resource shortlist | `asset-shortlist-v2.json` |
| Third-party checkout records | `third-party-inventory.json` |

The complete native PBR and optimization gates require further evidence. The terrain probe, measured local placement and local gallery leave the production Sim integration and real L5 gate independent.
