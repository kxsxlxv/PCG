# Stage 25 — coherent architectural archetypes

Scope: an opt-in SDK-free architectural comparison lane in research PR #23.
The 15 presets preserve the canonical three use IDs and regular cell ABI.
No previous stage, shader, or golden is removed. Selection is based on
stable building ID, resolved use, floors and explicit region with versioned
domain-separated SHA-256. Region weights (4:1) are illustrative priors,
not measured frequency or a source of GIS evidence.

## Reuse and design

- Reuse canonical L2 flat, gable and convex hip massing and L3 facade
  planner. Preserve exact footprint/holes, observed roof/eave and total
  source height. Inference failures may use flat with explicit reason and
  unchanged top; conflicting observed roof/eave requires review.
- Reuse PCG's audited OSM Building Grammar bay and oriented-box primitives,
  pinned upstream revision 81f1b50eae048645b2e620056a7cb2c84709bcfa,
  Apache-2.0; see lab/vendor/osm_building_grammar/UPSTREAM.md.
  Upstream overlays and Blender adapter are not adopted. No dependency added.
- Regular cell dimensions, deliberate blanks, facade zoning, window
  proportions, external balcony columns and coherent per-column finishes
  are declared per archetype. The core planner validates actual access,
  footprint exclusion and party/courtyard rules. Budget is reserved for
  whole balcony columns at conservative 20 triangles each, with rejected
  balcony fit falling back to windows rather than unguarded doors.
- The legacy plans retain their original digests. The optional variant
  parameter in the research tile compiler uses the same canonical plan.
  It exports the existing GLB role primitives, metric UVs, PCGF v1
  records and shared material bank. Compact face records use existing
  stage 12 research utility. No unique facade image per building.
- The provisional 1800 triangle cap covers host plus external balconies
  only. This experiment is not the final detailed LOD0 mesh or SDK adapter.

## Evidence and gate

Local: new planner unit tests cover all presets, source constraints,
geometry, deterministic selection, regional priors, complete columns,
party/courtyard and unchanged legacy digests. All 36 fixture plans also
pass local shell-wall matching. The full repository workflow is the
integration gate.

CI: run lab/zero_archetypes.py with the existing C++ PCGF decoder.
36 synthetic buildings: 15 equal-input comparisons by use, 15 different
dimensions/rotations and difficult input variants, six automatic selections.
Audit exact roof coverage, triangle count, aperture lookup, float32 packed
ABI, C++ decoder, actual external module overlap/clearance, GLB roundtrip,
reverse input-order invariance. Compare 1280x960 iso/front/night previews,
contact sheets and 1920x1080 neighborhood, rendered from reloaded geometry.
Publish report.json, plans.json, placements.json, tile.glb, packed binary
records, all PNGs and SHA-256 bundle.json in artifacts/zero-archetypes.
Actual CI and visual acceptance are reported only after run completion.

## Known scope limits

Synthetic presets are not calibrated on real GIS regional observations.
CPU diffuse and alpha visibility proxy is a deliberately isolated architectural
test; earlier GPU PBR/sidecar detailing, POM, roof equipment, widened paired
modules, private lobbies and native SDK are not integrated into this new lane.
Timber boards and curtain-wall appearance are shader proxies, not new measured
materials or geometric curtain-wall assemblies. Unchanged earlier research
lanes remain in CI. The current supported roofs are not a general concave
straight skeleton. Production adoption and hardware LOD thresholds require
separate verified integration, not an automatic switch of the GIS compiler.
