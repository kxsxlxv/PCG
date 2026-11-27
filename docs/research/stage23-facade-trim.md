# Stage 23 — facade construction, drainage and weathering

Stages 21 (final window material system) and 22 (entrances) are explicitly deferred while the licensed texture library is acquired. This consolidated experiment builds on stage 20 geometry plus the accepted stage 21A CPU appearance reference. No new commercial assets are included.

Previous visual gate: actual CI 54 close-up and brick facade reviewed again. Flat ground transition, repetitive facade plane, weak cornice profile, unmotivated noise and visually underdeveloped drain details. Window and lobby deficiencies deliberately remain out of scope.

Design: bevelled metric plinth drip profiles, restrained style-specific stringcourses and stepped lower cornice on flat-roof fixtures. Mitred rectangular corner joins. Existing gable cornice and slope soffit remain rather than adding overlapping moulding. Add one opposite-eave downpipe using the existing gutter geometry and an audited 6-sided sweep. No invented external drainage for flat roofs. Explicit opening clearance and conservative actual mesh accessory bounds cause local interruptions where needed. Weathering uses stable metric noise, rising damp, window-sill runoff, drain staining and underside deposits; it never alters aperture color/emission or cutout rail records. Geometry lane and shader-only lane use the same camera and shared baseline, with independent depth shadows and 1280 shadow maps.

Local preflight using CI 54 exported GLB, facade table and cells: 22 plinth segments, 24 belt segments, 8 cornice segments, 1 downpipe, 832 new triangles, 130 plan-view corner/contact pair checks. This is a geometry-planner result, not a current CI or full renderer result.

CI acceptance: verify GLB/PCGF readback, preserved original group prefixes and cell bytes, all module clearances, deterministic repeated plan, per-building triangle delta, unchanged opening colors/emission and cutout samples. FullHD baseline/geometry/shader-only images plus square plinth detail, back facade, aerial and night. Visual inspection is mandatory before accepting; green CI alone is insufficient. Report and standalone opaque GLB plus matching cutout GLB and ABI bundle in artifacts/zero-facade-trim. No merge to main.

Production follow-up: general polygon offsets, actual PBR roughness/normal and physical weathering masks, material/shader LOD and measured GPU timing in UNIGINE. The CPU color experiment cannot establish 2026 production photorealism.
