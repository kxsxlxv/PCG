# Zero-geometry facade research — checkpoint 01

- Exact tested head: `9573f02657feb552fe831bf7b2244bf1c632b312`
- [CI run 45709557839](https://github.com/kxsxlxv/PCG/actions/runs/45709557839), job 139540131671, SUCCESS.
- [Artifact 15066848550](https://github.com/kxsxlxv/PCG/actions/runs/45709557839/artifacts/15066848550), `zero-geometry/contact.png`, `report.json`, `merged_mesh.npz`, `parameters.json`.
- 49 core tests passed. 12 synthetic buildings, 156 triangles, 468 split vertices, 75 surface records. Eight role/material groups versus three shared role groups (logical counts, not draw calls). All 65536 uint16 indices round-tripped using two UNORM8 bytes; invalid bounds rejected.
- Identical depth across plain, day, night and denser opening layouts. Zero opening geometry. NPZ only, not a native mesh/GLB attribute roundtrip. Zone/per-floor triangle counts are analytical, not emitted.
- Actual preview inspected from CI JPEG log; roof and wall variations, coherent window grids and distinct night windows visible. Very flat illustrative windows, uniform grid, simple doors, repeated tiles, no photorealism. CPU orthographic diffuse and linear emission only, nearest sampling, no PBR/AA/mips.
- No SDK/GPU, no native draw-call measurements. Next: grouped material variation (macro breakup, grounded weathering, mask isolation, frequency diagnostics) on the same mesh, followed by night filtering and spatial stress. Keep research isolated; no merge.
