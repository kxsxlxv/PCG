# Stage 29 — private PBR replay and reference assets

Research-only. Stage 28 remains the immutable geometry/ABI baseline. The new
offline replay reads the original bundle, checks all SHA256 hashes and reconstructs
the existing renderer. It supports the 14 explicitly supplied source archives
without committing any binary data. 18 of 20 existing palette slots can be
overlaid, 3 as hybrid painted timber while retaining board seams; 2 standing
seam slots remain procedural. Source scale is estimated, not measured. Separate
maps and ARM/EXR inputs use explicit linear data channels, OpenGL normals,
linear-light albedo resampling, square tiling of rectangular maps, and metadata
repeat updates limited to compatible opaque records. Existing synthetic brick
cell tint is disabled for imported photographic bricks. Paving, peeling metal
and structured tiled facade are normalized but held out pending semantic mapping.

CI uses only original synthetic fixtures. A private replay must be run separately,
with the exact CI66 artifact extracted under artifacts/ and the user-provided
archives outside the checkout:

    PYTHONPATH=src python lab/zero_reference_assets.py --private-materials /path/to/archives --out /path/to/private-output

The output includes per-source checksums, normalized maps, overlay manifest, actual
baseline/candidate render pairs, and preserved stage-28 scene metadata. The
private output must not be uploaded to a public CI artifact or repository.
Source archive for the replay contains only already-tracked lab/src/approved assets.

Reference mesh intake: buildings_2012.zip includes 54 named mesh nodes and 14,924
triangles across mesh definitions; the separate glTF has 50 mesh definitions and
612,327 triangles across definitions, with inline geometry but separate PBR maps.
Three equipment families have three LODs each. They require EXT_meshopt_compression,
KHR_mesh_quantization and KHR_texture_basisu; existing uncompressed PCG importer
does not decode these. Do not copy donor models or infer licenses. AC LOD0 alone
exceeds the current whole-building 1,800-triangle cap. Geometry analysis and
original procedural variants remain gated on decoded geometry, placement checks
and budget-preserving integration. No change to source GIS, geometry or LOD in
the PBR replay. No native UNIGINE SDK claim.
