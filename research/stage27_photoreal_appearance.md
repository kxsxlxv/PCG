# Stage 27 — physically calibrated, use-aware appearance research

Status: implementation and local preflight; final CI and visual acceptance
are separate gates. The stage-26 renderer and all previous CI checks remain
enabled. This is a separate shader lane on the actual reloaded stage-26 GLBs.

## Goals and decisions

Stage 26's actual comparison and office detail showed washed-out brick,
over-visible repeated residential photo interiors in offices, uniform glazing,
canopies, railings and roof finishes. Keep all 15 archetypes and 36 buildings,
all attachments, exact GLB position/index/UV streams and original PCGF v1
bytes. No new opening triangles; near 22,148 and lite 13,838 triangles, max
1550/building. Vary appearance rather than change architectural silhouettes
at this stage, and use the exact previous cameras for an interpretable
comparison. Render all 47 previous views plus three office angle/day/night
probes. Keep the 0.95 near/lite silhouette gate.

Reuse pinned stage-26 commit 66306a17267f16cc6fc7a6b0ca18bb164a914a51:
existing GGX BRDF, metric POM and box ray functions, authored PCG material
recipes, Panelka and Three-Fenestra inputs already audited in the repo.
No new dependencies, third-party imagery, SDK, private assets or large
generated files committed. New original shared material layers (20) restore
brick hues, board cladding, standing-seam metal, roof tile, concrete and
stucco palettes using metric repeat and normal/ORM maps. Retain the previous
16 layer IDs. Map colors are explicitly sRGB; normals and ORM are linear.
Use deterministic archetype/building palette selection, age, window-frame
color, canopy/door material, and balcony panel styles. No claim that these
authored palettes are measured regional distributions.

Glass uses a Schlick effective F0 and two parallel incoherent dielectric
interfaces, R=2F/(1+F), T=(1-R)*transmission, with absorption/coating and
three parameter sets. Reflection is an approximate shared distant horizon
and existing procedural environment, not actual scene reflection or spectral
glazing. Interior radiance is reduced during daylight, canonical off/warm/
neutral night occupancy retained. Office and lobby interior rays use original
metric box surfaces/furniture and do not sample the residential atlas.
Residential approved imagery is sampled only on the rear surface; paired
door/window use the same room coordinates, key and occupancy state. Localized
sill streaks, ground dampness and clearance-aware panel seams replace arbitrary
wall-wide mottling. Curtain-wall spandrels are opaque, not invented windows.

Four balcony rail patterns (one legacy and three new) share six alpha texture
layers including roof and antenna. Preserve the actual cutout vertex/index/UV
streams; append only a building/kind appearance record and a uint16 face-record
sidecar. Correct alpha coverage mip by mip, never worse than unscaled at the
discrete cutoff, excluding antenna atlas scaling. Use repeat-U sampling to
avoid derivative discontinuities caused by fract in shader. This is a
research adapter, not a change to native GLB/PCGF ABI: a generic GLB viewer
still displays the stage-26 geometry/materials; stage-27 appearance needs
the new shader and shared banks. New material maps and masks go into CI
artifacts, not git.

## Local preflight and acceptance

Local checks used the actual CI #64 archived GLBs and source metadata, plus
original PCG generated maps and the reconstructed shared GLSL prefix/EGL
backend. 35,986 geometry ownership checks, 36 deterministic profiles,
150 wall records, 3832 occupied cells (2618 residential, 1214 office),
300 paired blocks, 12 lobby, 12 metal and 12 wood entrances. 56 appended
cutout records, 10,047 appearance records, 1302 cutout faces; exact
near/lite cutout remap. All twenty new PBR map recipes finite/in-range,
three new open masks with 17.55%, 19.38%, 23.96% coverage. Actual EGL
fragment compilation and 99-sample float framebuffer optical probe passed,
max FP16 error 0.0004831. All mip levels validated on actual prior masks.
No full repository clone locally (network restricted), no claim of full
local render/CI success. CI must render and verify exact bundle, all prior
regressions and new optical probe. Download and visually inspect before
acceptance. Native SDK, hardware performance, dense real GIS data and
actual geometry variation of canopy/balcony envelopes are separate gates.
