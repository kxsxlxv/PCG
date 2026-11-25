# PCG progress ledger

## L0.1 — historical probe CI
Status: implementation submitted; actual GitHub Actions result pending. This is infrastructure only, not acceptance of production geometry. Branch: lab/l0-visual-ci. Reproduce original geometry, architecture, asset, material, CPU CV, synthetic streaming, coordinate probes. Collect logs, JSON, GLB and PNG. Next: inspect run, fix failures, verify artifact download and actual PNG inspection. GIS audit restored but real dataset not included; synthetic audit fixture is a later L0 task. Do not advance L1 until L0 accepted. No SDK or GPU.

## L0 acceptance / L1.1
PR #1 merged as b19fbc870ad42e7e9ceedbbd4d4163044fbca76b (confirmed via GitHub). Prior review recorded run 45501693641, seven probes passing and actual architecture/material PNG review. L1.1 adds strict local metric rectangle input, typed immutable contract, canonical hash, fixture and unit tests. No geometry or new visual output yet; CI pending. Next: semantic shell and GLB with fixed-camera previews.

## L1.1 acceptance / L1.2
2026-11-25: PR #2 head ea5b44d90e264a0ea9391be39a65bfe5e020bf61, run 45533951198, 4 unit tests and 7 probes PASS; artifact 14986540562 metadata confirmed exact head. Nonvisual change, no new image review required. Merged and verified 2163fded4637c3f3d1009ded6dfe383194a9f45c. L1.2 adds analytic flat/gable semantic shell and topology tests; GLB and fixed-camera visual acceptance required before merge.

## L1.2 acceptance / L2.1
2026-11-25: PR #3 head f2adceda1e3d3a14cb25f04fbea3d08a66fe1100, run 45540229469: 5 unit tests (six shell configurations within test_shells) and 7 probes PASS. Artifact 14989046556 downloaded, SHA256 e612d99b98e791e6ac916519cdbfdef70293339f65dc2b29b20ce82b70ecb1fb matched. Opened flat_contact.png and gable_contact.png: all eight views show expected silhouettes and no obvious gaps; diagnostic triangulation lines intentional. Flat 12 triangles/576 m3, gable 16 triangles/672 m3. Merged and verified afa358410e2cc0581ac25512c22477ff80458ca7. L2.1 begins strict polygon footprint validation and canonical ring ordering; no geometry generation change yet. Invalid geometry rejected, repair policy deferred.
