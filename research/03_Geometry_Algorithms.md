# Research 03 — Computational geometry and building reconstruction

Date: 2026-11-24 | GeoWorld PCG | UNIGINE 2.22 | Sandworm excluded

Status: design and source audit with executed synthetic geometry experiments. No real Adler data or UNIGINE SDK was used; CGAL, Earcut, Surfer2 and Manifold were not installed or benchmarked in this environment. All proposed tolerances and thresholds must be calibrated. This document complements `01_Geospatial_Data.md` and `02_OpenSource_Analysis.md`.

## 1. Architecture and invariants

Offline pipeline: normalized BuildingDescriptor -> FootprintProcessor -> Massing / PartResolver -> RoofEngine / FoundationEngine -> SemanticGeometry -> Facade layout / module placement -> LOD compilation -> validation -> render mesh + instances + collision proxy -> exporter.

Maintain three representations: canonical GIS geometry (immutable reference and provenance), semantic surface/edge graph (metric double-precision, topology), and GPU mesh (triangles, local float positions, per-corner normals/UV/tangents/material IDs). A solid shell is for geometric validation; a render mesh may intentionally have split vertices at normals/UV seams, open decorative sheets, instanced details. Do not treat every GPU mesh as a manifold solid. Separate collision shell from roof trim and window decoration. Use stable building/part/surface IDs, deterministic per-feature random seed, recipe versions, local tile origin and explicit transform. Internal Z-up right-handed coordinates, metres; exporter explicitly handles winding, handedness, UV convention and unit conversion. glTF +Y up.

Never discard an otherwise usable building because a detailed roof failed. Preserve fallback reason and original desired geometry. Hard requirement: no NaN/Inf, invalid indices, zero-area triangles or unintended missing surfaces. Fallback must be visible in metrics. No dependence on Blender, Sandworm, Java or UNIGINE in runtime geometry semantics; use external offline tools where appropriate.

## 2. Footprint processing

Use PROJ, GEOS/Shapely for normalized metre-based operations. Use double precision and local origin before geometry work; preserve original geographic coordinates and vertical datum metadata. Validate finite coordinates, rings, consecutive duplicate nodes, short segments, intersections, hole containment and separation, connected components. Remove terminal duplicate only in open-ring internal representation. Normalize outer CCW and holes CW for Z-up; maintain edge metadata through reversal and reindexing. Extract only areal components from make_valid results; never silently select largest polygon. Quarantine ambiguous repairs. `buffer(0)` is not a universal repair. Use precision grids/robust overlay consistently rather than ad hoc rounding per backend. Avoid snapping across distinct buildings or across a courtyard.

Suggested starting ranges: 1–10 mm computational precision grid where appropriate, 2–5 cm short-edge candidate, 0.05–0.15 m near-collinear deviation, 1–3 degrees optional orthogonalization tolerance. These are not unconditional deletion rules. A short edge can define a real recess or a part seam; protect tagged edges, ridges and shared boundaries. Log area change, boundary Hausdorff distance, component/hole counts, adjacency changes. For valid original polygon target IoU >0.98 and boundary deviation <=0.15 m for near-LOD cleanup as a proposed criterion, with stricter exceptions for narrow features. Invalid original polygons need manual or independent reference comparison, not area ratio against signed invalid geometry. Detect near-zero areas, slivers, repeated nodes, acute reflex corners; allow multipolygon with deterministic ordering.

OBB / minimum-area rectangle and long-edge orientation are useful features, not permission to replace a footprint. Rectangular fast path only if no holes, convex, near rectangle in metric space, high overlap, acceptable Hausdorff distance and edge lengths, compatible roof metadata. Preserve footprint or use exact quadrilateral algorithm when possible. Near-square orientation is ambiguous: prefer known ridge direction, adjacent buildings, stable seed. Do not impose Manhattan grids on non-orthogonal architecture. For decomposition, use reflex-vertex candidate cuts/orthogonal decomposition, score compactness, orientation, internal seam count and fit; avoid treating arbitrary convex partition as physical building segmentation. Part metadata overrides geometric guess. Compare exact parts against residual parent area; render parent only where policy and evidence warrant (not as a second full volume).

## 3. Triangulation

Default GEOS constrained Delaunay triangulation (Shapely >=2.1 / GEOS >=3.10) for Python MVP. C++ compare GEOS CDT, CGAL CDT (mark interior domains), mapbox Earcut. Earcut is a fast candidate, not an automatic repair or correctness guarantee. For triangles use direct emission; convex quads choose diagonal by quality and semantic constraints. For planar surfaces triangulate in local 2D surface frame then lift back to 3D. Include hole rings, shared edges, ridge constraints. Do not triangulate a concave roof projection with ordinary Delaunay and discard outside triangles: may leave gaps. Use constrained triangulation of individual planar roof faces or PSLG, preserving all semantic breaklines. Test union coverage and overlap; area sum alone is not enough. Add boundary-segment conformity checks and avoid T-junctions.

Executed probe (geometry_probe.py): 250 deterministic valid radial polygons, seed 7321, 6–20 vertices. GEOS 3.13.1 / Shapely 2.1.2. CDT zero missing/outside/overlap in test precision (see report for selected named cases); ordinary Delaunay + covers filter lost area for 96/250, max 50.053563866 m². CDT produced 2735 triangles, filtered method 2568. Elapsed times are run-dependent and include Python validation, not a pure library benchmark. Named fixtures include rectangle, L, U, T, courtyard, narrow, irregular, notch, near-collinear, bowtie (repair only). Bowtie make_valid returned two polygons, total area 32 m². No real-world coverage claims.

## 4. Roof model and height resolution

Internal Z values relative to canonical ground reference: base_z, eave_z, roof_top_z; total height from ground to top, min_height is relative lower level rather than thickness. For gable end wall the wall top follows the roof profile (eave_z is a reference, not every top vertex). Keep source height reference (ridge/eave/mean/unknown), estimated height uncertainty and whether roof pitch/height is observed or inferred. Never add roof height to OSM total height again. Constraint solver should resolve Htop >= Heave >= Hbase; floor heights, ground floor, attic, parapet, foundation independently. If angle and height disagree, prioritize source quality and report conflict. Never shrink an explicitly measured body height silently.

### Analytic models

Local roof frame: x along ridge, y across. For rectangle of length L and width W, centered y in [0,W], equal pitches alpha: z=eave + tan(alpha)*min(y,W-y); rise = W/2*tan(alpha). For 12x8 m and 30 deg, rise=2.3094 m. If rise explicitly given, alpha=atan(2*rise/W). Ridge orientation normalized modulo 180 degrees; downslope direction is directional (modulo 360). For skillion use affine z over projection coordinate, normalized to a specified rise. For convex polygon with all edges sloped equally, roof height = eave + min_i(tan(alpha)*signed_distance_to_supporting_line_i); valid because all inward halfplanes define convex polygon. Does not generalize to nonconvex polygons. Weighted convex case uses per-edge pitches. Partition roof cells by active minimum plane, not all pairwise plane intersections in production. Our proof-of-concept splits all plane equalities, causing excess triangles (rectangle hip 54 total rather than minimal shell). For concave gable a single two-plane field clipped by footprint is geometrically possible but may yield implausible high eaves on wings; choose architecture-aware subdivision / weighted skeleton instead. For pyramidal roof use true apex if appropriate (square equal-slope hip collapses to point), explicit apex fan only for star-shaped polygons with visible apex, otherwise avoid.

### General straight skeleton

Inward-moving edges create wavefront, vertex trajectories form skeleton. Edge collapse and reflex vertex split events; holes interact too. Distinguish straight skeleton from medial axis (latter may have parabolic arcs; different topology). Unweighted uniform slope; skeleton event offset t maps to z=eave + t*tan(alpha) for unit-speed offsets. Weighted edge speed proportional to cot(alpha) if vertical rise used as time. CGAL weight convention must be handled by documented angles API; do not substitute mathematical speed directly without checking. CGAL weights ordered by edge ending at ith vertex (previous -> current); preserve mapping after reversing rings. Vertical edge in `extrude_skeleton()` may use weight 0; weighted skeleton construction is a different API and its weights have stricter preconditions. CGAL supports holes, weighted offsets and capped extrusion. Use `CGAL::extrude_skeleton(pwh, mesh, CGAL::parameters::angles(...).maximum_height(...))` as documented example, NOT compiled here. Angles are degrees per API; use a pinned version and test edge indexing. If height limit truncates slopes it changes morphology (flat crown), not equivalent to scaling the rise. Preserve requested roof topology when reconciling height. Avoid using extruded bottom face as visible roof. Prefer skeleton face provenance (originating contour edge) or recover semantic IDs from backend; generic triangle output alone is insufficient. Validate roof footprint, ridge continuity, closed wall contact, slopes, drainage, no degenerate/self-intersecting faces, rise and overhang. Bound per-building CPU/memory and isolate backend failures. Retry only with logged acceptable simplification; use analytic only when compatible; flat fallback with diagnostic reason.

### Roof families / planned backends

- Flat: CDT + optional parapet (offset ring, robust boolean); roof drainage detail for close views optional.
- Gabled: rectangle analytic or clipped two planes; weighted skeleton vertical gable edges; complex wings via decomposition and union.
- Hipped: convex analytical or CGAL for concave and holes.
- Half-hipped: truncated gable end geometry, explicit breaklines; verify metadata convention.
- Skillion: one plane, variable wall-top heights, slope direction distinct from ridge.
- Pyramidal: hip or apex construction for suitable footprints.
- Mansard: two slopes, intermediate offset profile, upper offset can split/vanish, handle topology changes. Gambrel: two-pitch gable cross-section with end caps. Saltbox: asymmetric gable, solve ridge from slopes and span; source height constraints.
- Sawtooth: repeated sectional profile for industrial halls, clerestory vertical faces and UV.
- Round/barrel/dome/onion: parametric surface tessellation on compatible plan, radial/adaptive segments; MVP optional; no blind fitting to arbitrary outline.
- Explicit OSM ridge/edge metadata: preserve as constraints, derive planes or use reference backend (check feasibility).

Prioritize flat, gabled, hipped, skillion and parapets; then mansard/half-hipped/pyramid, industrial profile, specialized roofs. Actual Sochi frequencies not measured. Regional style inference decides unknown roof family, not geometry alone.

### Overhangs

Do not inflate wall footprint to roof edge. Offset roof footprint via miter-limited offset; holes shrink for inward overhang into courtyard; check collapsed holes and self-intersections. Preserve contact on original wall ring. If eaves extend across slope by d, z drops d*tan(alpha) on sloped side, not at gable-side horizontal extension. Use explicitly generated fascia/soffit close up; avoid two coplanar opposing surfaces in solid validator. Far LOD may omit overhang only when projected error is small. Handle adjacent/party walls, trim overlap, overhang not penetrating neighboring buildings. Distinguish source roofprint from footprint before adding overhang.

### Source audit

- Condor gabled source uses oriented bounding box and quadrilateral/no-hole assumptions, roof and gable walls separate, lowered eave extension and double-sided slope triangles. https://raw.githubusercontent.com/jlgabriel/condor-buildings-generator/main/condor_buildings/generators/roof_gabled.py
- Condor polyskel backend uses embedded bpypolyskel + mathutils; standalone import failure falls back flat. https://raw.githubusercontent.com/jlgabriel/condor-buildings-generator/main/condor_buildings/generators/roof_polyskel.py
- OSM2World HeightfieldRoof uses roof constraints and height evaluation/interpolation, supports holes in representation; examine getInnerSegments, FaceDecompositionUtil and sloped UV mapping. https://raw.githubusercontent.com/tordanik/OSM2World/master/core/src/main/java/org/osm2world/world/modules/building/roof/HeightfieldRoof.java
- OSM2World GabledRoof derives AbstractGabledRoof. https://github.com/tordanik/OSM2World/tree/master/core/src/main/java/org/osm2world/world/modules/building/roof
- osm2city uses roof hints, contextual selection and simplification. https://osm2city.readthedocs.io/en/latest/how_it_works.html ; https://osm2city.readthedocs.io/en/latest/parameters.html
- CGAL https://doc.cgal.org/6.2/Straight_skeleton_2/group__PkgStraightSkeleton2Extrusion.html ; https://doc.cgal.org/6.1.3/Straight_skeleton_2/group__PkgStraightSkeleton2WeightedSkeletonFunctions.html ; https://www.cgal.org/2023/05/09/improved_straight_skeleton/
- Surfer2 alternative exact-arithmetic weighted PSLG solver, independent comparison; Monos specialized monotone polygons. https://github.com/cgalab/surfer2 ; https://github.com/cgalab/monos ; https://www.cosy.sbg.ac.at/~held/projects/wsk_exact/wsk_exact.html
- Python alternative https://github.com/iconbuild/py_straight_skeleton (holes documented), https://github.com/prochitecture/bpypolyskel ; https://github.com/botffy/polyskel. Not tested here.
- Roofer (only if useful point clouds) https://github.com/3DBAG/roofer.

## 5. Building parts and 2.5D arrangement

Normalize part polygons and their vertical intervals, keep parent id and ownership. OSM building outline generally represents entire building; part data must not be added to full parent extrusion. Explicit vertical gaps (min_height) represent raised parts, passages, bridge connections; do not extrude columns to ground by default. Use common building ground reference across parts, not independently sample terrain for every section. If parts cover only some parent footprint, distinguish incomplete mapping from intentionally open areas; residual policy must log uncertainty; never cover a courtyard. Resolve 2D overlap with vertical occupancy, not simply union all footprints at tallest height.

For flat-roof prism groups: sort unique z endpoints; for each slab [z_i,z_{i+1}], union all active footprints U_i. Vertical wall boundaries = exterior and hole edges of U_i over slab. At each level z_i, upward exposed faces = U_below minus U_above; downward exposed faces = U_above minus U_below. Empty layer outside interval; interior coincident faces omitted. Assign original part/material ownership from planar overlay. Node common XY boundaries across bands to avoid T-junctions; weld topological vertices at seams. Merge coplanar faces only if role/material/UV semantics permit; split complex output into planar polygons with holes. Point/edge touching separate solids may create nonmanifold union: classify connectivity and keep separate shells or resolve tolerance explicitly. Do not use this slab approximation for inclined roof intersections. For complex slopes/volumes, use local robust mesh booleans on closed, oriented inputs, or exact planar roof arrangement and seam clipping. CGAL PMP corefinement with exact constructions for repeated operations; Manifold as alternative candidate with valid manifold inputs, verify face provenance and numerical behavior. CSG for structural junctions only, not each window across an entire city. Sources https://doc.cgal.org/latest/PMP_Boolean_operations/ ; https://github.com/elalish/manifold ; https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings.

Executed slab test: 3 prisms (overlapping lower/upper towers and floating part). Naive sum 1376 m³, union 1184 m³. At 6 m and 12 m upward exposed areas each 64 m²; at 8 m downward exposed area 16 m² (overhang underside); at 10 m upward area 16 m². At 0 m downward 128 m². Only occupancy and horizontal exposure tested, not a full stitched multi-part mesh. Elevated floating part is a synthetic test, not assertion that unsupported geometry is architecturally valid.

## 6. Terrain foundations

Use the same authoritative terrain sampling system and vertical datum as runtime. DTM coarse raster resolution is not improved by interpolation. Sample footprint interior, contour, vertices, and adaptively subdivide edges when terrain interpolation error exceeds tolerance or segment too long relative to reliable data. For each building choose horizontal datum using entrance/road/known ground floor where available; otherwise robust interior/perimeter quantile with explicit policy (e.g. high quantile for avoiding uphill wall burial, median for mild slopes). Do not tilt houses. Preserve total height reference, adjust foundation separately. Keep walls horizontal by floor; terrain-adaptive skirt from base to actual ground, bury small tolerance; where terrain higher than base, use partial buried wall / stepped foundation or local site grading, not negative skirts. Do not automatically flatten whole terrain tile. Severe slope requires retaining walls/terraces or split massing; slope thresholds are data- and archetype-dependent. Underpasses/raised parts exempt foundation rules; use explicit supports or validated rules, no solid fill. Shared contact geometry for neighbors, deterministic across tiles and terrain LOD. Separate ground collision from building shell; assess floating gaps, excessive burial, foundation heights and transitions. DTM uncertainty can exceed visual tolerance, report independently. Test synthetic linear slope, saddle, discontinuity, no-data, varying terrain LOD, coastal slope.

## 7. Semantic surfaces, UV and geometry detail

Representation:
```
SemanticSurface { stable_id, part_id, role, polygon_3d_with_holes,
  plane_or_parametric_patch, local_frame, material_slot, edge_ids,
  source_edge_id, facade_layout_id, lod_visibility }
SemanticEdge { stable_id, endpoints, incident_surface_ids,
  role: ridge|valley|hip|eave|gable|party_wall|foundation|seam }
```
Use halfedge/B-rep-like connectivity for shell; render mesh per-corner attributes. Preserve UV seams and hard corners, not weld by position alone. Facade basis tangent along wall and up; u=dot(p-origin,t)/tile_width, v=(z-z_floor_reference)/tile_height. Lay out bays from actual facade length in metres, left/right margin, integer window count and floor heights; do not stretch a whole facade photo. Keep window and floor layout independent of triangulation. Sloping roof basis: normalized in-plane eave direction u, v = n cross u (choose sign consistently, along uphill if desired). UV = dot(p-origin,u)/tile_m_u, dot(p-origin,v)/tile_m_v. For pitch alpha, true slope length = horizontal run / cos(alpha); XY projection compresses texture repeat in slope direction. Valleys/ridges can intentionally have UV seams but scale and material direction must remain consistent. Eaves, fascia, gutters and roof details use actual edge graph. Tangents MikkTSpace when compatible with UNIGINE normal map pipeline; hard normals on wall and roof plane transitions, smooth only curved surfaces; validate tangent sign and orientation after export conversion. Use shader/geometry mix: near openings may be recessed via local facade panel decomposition rather than global mesh booleans, mid relief geometry normal map, far texture. UV seams in rendering are not shell cracks.

Sources: https://github.com/mmikk/MikkTSpace ; https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html ; https://www.cityjson.org/dev/modelling-guide/ .

## 8. LOD generation and geometry budget

Separate GIS LoD1/LoD2 semantics from rendering LOD0...3. Generate all levels from one canonical shell + architecture recipe/seed. LOD0 shell + instanced balconies, windows trim, parapets, roof fixtures; LOD1 retain silhouette, simplify protrusions/merge detail, LOD2 retain main roof planes and massing with baked/parametric facade; LOD3 simplify footprint/roof only when projected error allowed, not flatten all roofs by default. Important landmarks preserve signature shapes longer. Simplify semantic polygon boundaries first, preserve holes if visible, ridge/valley, interfaces, vertical envelopes. Rebuild mesh per LOD and use meshoptimizer for indexed geometry/cache/overdraw/fetch optimization and selective simplification with attributes/protected discontinuities. Avoid material/UV boundaries being collapsed. Do not confuse triangle count reduction with guaranteed frame time. Screen-space geometric error e_px≈epsilon_m * H_px/(2*d*tan(FOV_y/2)), using view depth / conservative bounds; hysteresis and near clipping cases; tune in engine. Example 0.4m geometric error at 500m, 1440px, 60deg vertical FOV ≈1 pixel. Record measured error, triangles, vertices, material surfaces, instances, collision proxy, bounds. Parent HLOD later in rendering research. Proposed simple budgets (not tests): 12–50 tris for basic distant shells, 24–150 for medium simple shells, 100–500 for closer shell/details, 500–2500 for near typical buildings including details; large complexes differ. Prototype hip split is inefficient and must be replaced.

https://github.com/zeux/meshoptimizer ; https://github.com/zeux/meshoptimizer/blob/master/src/meshoptimizer.h

## 9. Interfaces and implementation sequence

```
FootprintProcessor::normalize(raw) -> CanonicalFootprint + RepairReport
PartResolver::resolve(building) -> MassingGraph + ConflictReport
IRoofBackend::build(RoofRequest) -> RoofPatchGraph | Diagnostic
FoundationEngine::build(massing, ITerrainSampler) -> FoundationSurfaceGraph
FacadeLayout::build(wall, style, seed) -> BayLayout + ModuleInstances
MeshCompiler::compile(semantic, RenderLOD) -> MeshBundle + ValidationReport
Exporter::write(bundle, tile_transform) -> artifacts + manifest
```
RoofRequest includes polygon with holes and stable edge IDs, observed shape/angles/ridge, requested eave/rise and priority, overhang policy, neighbor constraints. RoofPatchGraph includes plane/patch, boundaries, face role, source edge ID, heights, roof coverage, status/backend/fallback/elapsed. Use version-pinned CGAL C++ worker (batch requests, timeout/isolation), no Blender in runtime. Start with Python Shapely+NumPy proof of concept; avoid premature C++ reimplementation before fixtures. Production path C++ GEOS/CGAL/Earcut/meshoptimizer; choose exact versions after local build. Implement flat+prism first; analytic rectangle/convex; CGAL complex and holes; 2.5D parts; terrain; UV + semantic export; LOD and budgets; then grammar integration. Maintain deterministic order, per-building independent failures and diagnostics. Do not implement an entire general straight skeleton yourself.

## 10. Validation and benchmark protocol

1. Input: geometry validity, explicit repaired status, area/hole/part invariants, bounds, unit/datum checks.
2. Planar: triangulation union covers domain with no overlap/outside beyond tolerance; constraint edges present; verify holes remain open; seam noding.
3. Shell: oriented triangles, edge incidence 2 for each closed shell edge, vertex link manifold check, positive volume, self-intersections checked separately, no unexpected disconnected components, roof/wall contact, no zero area. Watertight alone is not proof of a valid solid. `CGAL::Polygon_mesh_processing::does_self_intersect` and https://github.com/tudelft3d/val3dity (CityJSON/ISO19107) candidates; trimesh basic checks are not full validator.
4. Semantic: roof height, pitch, footprint area, material assignment, UV continuity/scale, source edge provenance, repeated floor alignment, no duplicate parent and part geometry.
5. Terrain: clearance, penetration, foundation extent, raised passage preserved. Rendering: LOD silhouette, shader UV, seams, shadow proxies and collision.
6. Repeatability: canonicalize ordering and random seeds; compare output hashes on pinned platform, use tolerance-based geometry comparisons across kernels/platforms. Parallel batch with stable IDs, deterministic tile ownership.
7. Benchmark actual Adler after dataset audit: stratify geometry (rectangular, concave, holes, high-rise, parts, steep terrain); measure per-stage P50/P95/P99, worst cases, fallback by type, triangles/vertex/material counts, peak memory, throughput with worker counts, review failures. Separate generation latency from UNIGINE runtime performance. Suggested acceptance: 100% valid accepted/fallback shells on approved fixtures, no lost courtyard or duplicate volume, all rejected/repaired counted, no hidden fallback. Complex roof success goal >=99% on locally verified eligible polygons is target, not established; calibration needed.

## 11. Executed artifacts

`geometry_probe.py` generates analytic flat, two-plane gable and convex hip models, performs constrained triangulation comparison and basic solid metrics. `geometry_probe_results.json` contains exact last run output. `geometry_probe.glb` contains 20 geometry-only colored test volumes, positioned in a grid; exported +Y up and reloaded using trimesh. No texture UVs, real facade, roof overhang, skeleton, sloped foundation, full multipart stitched shell, LOD chain or UNIGINE integration. GLB render copy splits vertices intentionally; manifold checks are on welded core shell before export. GLB is diagnostic, not production asset pack. Test results: 20/20 watertight, consistent winding, positive volume, no degenerate faces; max volume error vs integrated roof height 3.5e-7 m³. This does not prove absence of self-intersections (not tested) or architecture correctness. Fixed global X ridge in gable probe; complex gables may look unrealistic. Convex hip uses inefficient all-plane equality subdivision. No CGAL installed and no actual Adler geometry. Re-run:
```
python -m pip install 'shapely>=2.1,<3' numpy trimesh
python geometry_probe.py --output geometry_probe_results.json --models geometry_probe.glb
```
Versions: Shapely 2.1.2, GEOS 3.13.1, trimesh 4.11.1. Timing changes between runs. Source links are branch-based and must be pinned to commit/version by Codex.

## 12. Decisions and open questions

Adopt CDT for prototype; keep Earcut fast path subject to benchmark. Use analytic fast path and CGAL weighted skeleton as main candidate, compare Surfer2 if necessary; no universal skeleton for every roof. Use semantic geometry before render triangles, 2.5D arrangement for flat parts, robust CSG only for local true 3D intersections. Use terrain-conforming foundations, preserve actual horizontal massing and vertical datum. No full procedural architecture yet. Next research 04: regional grammar, constrained facade layout, semantic hierarchy, reproducible asset rules and automated visual quality. Research choices, not user-imposed: C++ worker, Python first, thresholds, fixtures, LOD tiers and backend order. Need actual SDK, local data, build and test complex skeleton + parts before final stack selection.
