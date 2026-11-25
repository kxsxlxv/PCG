# PCG progress ledger

## L0.1 — historical probe CI
Status: implementation submitted; actual GitHub Actions result pending. This is infrastructure only, not acceptance of production geometry. Branch: lab/l0-visual-ci. Reproduce original geometry, architecture, asset, material, CPU CV, synthetic streaming, coordinate probes. Collect logs, JSON, GLB and PNG. Next: inspect run, fix failures, verify artifact download and actual PNG inspection. GIS audit restored but real dataset not included; synthetic audit fixture is a later L0 task. Do not advance L1 until L0 accepted. No SDK or GPU.

## L0 acceptance / L1.1
PR #1 merged as b19fbc870ad42e7e9ceedbbd4d4163044fbca76b (confirmed via GitHub). Prior review recorded run 45501693641, seven probes passing and actual architecture/material PNG review. L1.1 adds strict local metric rectangle input, typed immutable contract, canonical hash, fixture and unit tests. No geometry or new visual output yet; CI pending. Next: semantic shell and GLB with fixed-camera previews.

## L1.1 acceptance / L1.2
2026-11-25: PR #2 head ea5b44d90e264a0ea9391be39a65bfe5e020bf61, run 45533951198, 4 unit tests and 7 probes PASS; artifact 14986540562 metadata confirmed exact head. Nonvisual change, no new image review required. Merged and verified 2163fded4637c3f3d1009ded6dfe383194a9f45c. L1.2 adds analytic flat/gable semantic shell and topology tests; GLB and fixed-camera visual acceptance required before merge.
