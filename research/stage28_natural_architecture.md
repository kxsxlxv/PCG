# Stage 28 — reducing procedural repetition with bounded geometry

Status: implementation and local preflight. CI, downloaded artifact and actual
visual review are separate acceptance gates. Preserve all earlier CI lanes.

## Scope and implementation

Prior actual stage-27 comparison_details: visibly periodic plaster/panel
microtexture, regular brick, simple office furniture, uniform glass reflections,
dark glazed balconies and generic entry slabs. This increment uses the same
36 synthetic buildings, all 15 archetypes, canonical plans, original PCGF
cell bytes and paired opening metadata. Keep all existing geometry and
alpha-tested guard/antenna GLBs. Add only external, fitted attachments using
the pinned stage-23 metric prism and stage-26 compact append/export APIs.
Reserve budget for each entrance and full balcony column before allocating
optional sill groups in deterministic order. Near max 1800 triangles per
building, global additional limit 9000. Lite retains entrance caps and balcony
edges but omits projecting sills. Existing slab/guard envelope and threshold
remain unchanged. Source vertex/index/layout prefixes and the entire previous
PCGF record payload (excluding its changed record-count header) must match.
No per-opening geometry on curtain walls, paired balcony blocks or internal
courtyard/party walls. Conservative 2D convex hull + Z envelope tests against
real source attachments and the host footprint; exact facade opening rectangles
checked, no claim of general mesh-intersection/structural certification.
Plan rejects an entire column/bay if it does not fit the allocated budget.

Reuse pinned stage-27 commit 22c1f68a4f09f99aef899c524b284efd41f4c1b3:
existing original PCG material recipes, audited Panelka/Three-Fenestra inputs,
GGX, POM, ray-box functions and EGL lab; no new dependencies or external imagery.
Six plaster/concrete maps replace sinusoidal periodic components with seeded
isotropic FFT band-limited noise. Shared sources remain periodic and two-phase
smooth shader blending reduces repetition; structured brick/wood retains the
bond/board direction and per-unit color variation. Age is continuous in metric
building coordinates, not arbitrary independently phased walls. Distant PBR
paths omit normal and secondary texture samples, and simplified room paths
omit furniture intersections. No new texture layers, all 36 layers retained.
Use SRGB8_ALPHA8 albedo and RGB8 normal/ORM instead of FP16: actual GL format
queries and 1728 shader texel samples compare to the unquantized inputs.
Logical PBR-array storage (full mip chain, not total VRAM) halves from
251658000 to 125829000 bytes. Room, alpha, environment and metadata storage
unchanged except new opaque/neighbor metadata; no native timing claim.

Reflection uses existing procedural environment plus direction-continuous
cloud contrast and four actual nearest canonical neighbor AABBs per building.
These are shared small tile metadata, not one reflection texture per building;
bounding boxes are approximate, not exact reflected facade geometry, SSR,
HDRI or GI. Independent plane-intersection oracle and 32 actual shader rays
include parallel hit/miss cases. Retain the stage-27 three effective glass
profiles. Rooms retain semantic routing and shared paired-room coordinates:
furniture variations, desks/screens/keyboard/chairs, meeting tables, cabinets,
lobby fixtures and residential rear atlas. No residential imagery in offices.
Subtle balcony bounce is derived only from existing occupancy state; no
additional room state or real light source.

## Acceptance

Local preflight against actual CI #65 archives, reconstructed prior exporter
and EGL backend (not full repository clone): 36 canopy caps, 263 balcony
front-edge profiles, 358 projecting sills; 8 profile variants; 282846
conservative aperture/envelope checks; 8676 added triangles, near 30824,
lite 18218, max 1798. Budget-rejected candidates recorded (2161 optional
candidate modules and four aperture conflicts; not missing host geometry).
Exact reloaded GLB prefixes and cutout face index remap; 14128 opaque records,
14187 appearance records, reversed input unchanged. Six spectral peak
fractions decreased from 0.8464–0.9279 to 0.2883–0.3283. Local shader
compilation and 1728 texel samples passed, max error 0.001637; 32 reflection
ray samples, max FP16 error 0.000938 m. Local selected render harness used
fallback room atlas and no scene shadows; final open CI uses full audited
inputs. CI rerenders all 50 previous cameras plus six matched geometry-only
views, checks C++ ABI, exact day/night foreground masks and >=0.95 near/lite
mask IoU. Download the actual archive, verify all manifest entries and
inspect previous and new views. Report known defects and actual status in PR.
Stage 28 is SDK-free research; no native UNIGINE integration, real-GIS regional
calibration, hardware performance measurement or private textures.
