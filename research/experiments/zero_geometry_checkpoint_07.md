# Checkpoint 07: approved synthetic GIS bridge

Head `39b04ae539c9cdfb7edd80898cb5a7ece5ac5407`, [CI 45715888997](https://github.com/kxsxlxv/PCG/actions/runs/45715888997), PASS all workflow steps and 49 core tests. Artifact 15070581117.

Existing approved pilot report -> two actual zero-opening mesh tile exports. The source is synthetic, not a real Adler survey. Original detailed meshes 9,776 triangles; preserved semantic shells 48 triangles (16 house, 32 office) and all 128 canonical opening intents (33/95). Same plan digests, footprint, roofs and triangle coordinate matches within 2.95 micrometres in the fixture's local frame. Both independent C++ ABI checks passed, 12,288 numeric lookups checked. Sampled DTM anchors and source lineage preserved; one unresolved record, one unapproved group and building parts retained in exclusion report. No foundation construction. The detailed and shell meshes do not have equivalent physical facade details, so the reduction is not a same-quality performance benchmark.

Next: actually export 1,024 buildings with varied heights, verify disk packages independently (hashes, paths, ABI, indexed geometry, unique ownership), and reject a corrupted payload. CPU time and process RSS only.
