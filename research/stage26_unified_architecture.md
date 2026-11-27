# Stage 26 — unified architectural detail research

Status: implementation and local preflight; CI and visual acceptance are separate gates.

Integrate all 15 stage-25 archetypes and 36 canonical plans with the existing
roof, external balcony, window, alpha-cutout and EGL PBR research components.
The stage-25 canonical cells and host triangles must remain unchanged, including
the GIS footprint and roof height. Detail envelopes intentionally extend beyond
the host. All 300 balcony anchors remain external, with opening pairs stored
in an explicitly versioned research sidecar rather than changing PCGF v1.
Use original shared metric materials and the audited existing open material
sources, no additional third-party assets or private EXRs. Preserve the legacy
GPU lab and all existing CI checks. The production GIS compiler is not switched.

One deterministic planner emits complete detail modules, conservative collision
envelopes, exact roof support and attachment checks, and explicit rejected
placements. Near and lite use the same host, aperture metadata, balconies and
primary detail modules; lite removes complete secondary modules. The 1800
triangle cap includes actual accessories and alpha-cutout faces per building.
Recompile the source host and verify exact GLB position, index, color and UV
prefixes; independently reload near/lite opaque and cutout GLBs. Verify the
legacy C++ ABI and compare float32 GPU opening lookup with an independent
rectangle oracle. Render the same 36 buildings in EGL software PBR and inspect
matched stage-25 comparisons, close facade and roof views, day/night and
near/lite neighborhood. The neighborhood foreground-mask IoU gate is 0.95.

Local preflight against the actual archived stage-25 plans and host GLB:
36 buildings, 300 external balcony anchors, 120 glazed, 300 paired blocks,
35,867 conservative envelope checks, 22,148 planned near and 13,838 lite
triangles, maximum 1550. Two roof equipment placements rejected due to
unsupported clearance. Metadata preflight: 4345 canonical cells, 4495 opening
rows, 9988 opaque records, 157768 lookup samples, maximum paired float32 edge
drift 1.526e-6 m. Full repository clone and end-to-end EGL test are CI gates,
not claimed local successes.

Limitations: synthetic regional priors; SDK-free software rendering, no
hardware timings or native UNIGINE integration; paired sidecar still needs
native ABI support; lite is not an empirically measured far-LOD threshold.
Timber and curtain-wall appearances are shader proxies. No general concave
pitched roof solution, structural certification or dense-city adjacency proof.
