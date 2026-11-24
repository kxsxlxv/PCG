# Incremental roadmap (one task / PR / review)

Do not give Codex the entire system as one task. Each increment must build/run in CI, leave artifacts, document what is not solved. Human review for architecture scope and visual baseline changes. No auto merge or broad refactor.

| Gate | Bounded deliverable | Evidence / stop condition |
|---|---|---|
| L0 Visual Lab bootstrap | Restore historical probes 01/03/04; run geometry+facade+material+coordinate+streaming checks on standard CPU runner, upload PNG/GLB/JSON/logs | Actual successful workflow and retrieved/inspected image; if connector transfer blocked, document manual path. No production modules yet |
| L1 contract + one house | Python package skeleton, schemas, explicit metric rectangle, flat/gable, deterministic semantic surface graph and GLB | unit tests, top/side/iso render and closed shell |
| L2 geometry fixtures | L/U/T/courtyard, invalid repair, convex hip, documented fallback, pitched roofs and holes, actual tests | geometry invariants, visual grid; no full general skeleton claim |
| L3 facade/asset integration | 3 representative profiles, floor/bay plans, correct aperture and module anchors, LOD shared plan | bounds/overlap/party-wall tests and close/roof views |
| L4 basic PBR + visual block | original materials, metric UV, day/night mask, fixed camera neighborhood, optional CPU Blender lane | geometry and visual comparisons; no GPU equivalence claim |
| L5 GIS + terrain contract | real authorized snapshots only via explicit local input, source audit + identity fusion, DTM sampling and rejected/fallback log | real small Adler sample, manual independent reference subset |
| L6 complex geometry & tiles | general skeleton backend, parts/foundations, LOD error, spatial ownership, neutral manifests, simulated scheduler | stress fixtures, HLOD ownership and deterministic rebuild; no native performance |
| N0 SDK smoke | G0–G8 local SDK integration checks, minimal packer | actual compile/render/screens and shader/axis tests |
| N1 native pilot | G9–G12, spatial batches/cluster, bounded streaming, profiling | measured workstation logs, accepted visual/functional quality |
| N2 extension | larger Sochi region, optimized materials, optional pretrained enrichment | measured incremental gain; no training |

L0 is infrastructure and reproduction, not product implementation. Research probes must remain clearly labeled. Start L1 only after L0 evidence reviewed. Parallel work on offline core allowed once schemas stabilized; native integration does not block offline development. No date/effort estimates without measured complexity. 
