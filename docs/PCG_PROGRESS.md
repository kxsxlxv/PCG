# PCG progress ledger

## L0.1 — historical probe CI
Status: implementation submitted; actual GitHub Actions result pending. This is infrastructure only, not acceptance of production geometry. Branch: lab/l0-visual-ci. Reproduce original geometry, architecture, asset, material, CPU CV, synthetic streaming, coordinate probes. Collect logs, JSON, GLB and PNG. Next: inspect run, fix failures, verify artifact download and actual PNG inspection. GIS audit restored but real dataset not included; synthetic audit fixture is a later L0 task. Do not advance L1 until L0 accepted. No SDK or GPU.

## L0 acceptance / L1.1
PR #1 merged as b19fbc870ad42e7e9ceedbbd4d4163044fbca76b (confirmed via GitHub). Prior review recorded run 45501693641, seven probes passing and actual architecture/material PNG review. L1.1 adds strict local metric rectangle input, typed immutable contract, canonical hash, fixture and unit tests. No geometry or new visual output yet; CI pending. Next: semantic shell and GLB with fixed-camera previews.
