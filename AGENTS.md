# AGENTS.md — PCG contributor instructions

Read docs/ARCHITECTURE.md, DATA_MODEL.md, TECHNICAL_SPECIFICATION.md, IMPLEMENTATION_ROADMAP.md, TESTING.md and CI_VISUAL_LAB.md before code changes. docs/ is normative; research/ holds evidence/probes, not production implementation.

- Work on ONE bounded increment/issue at a time, in a branch and PR. The user authorizes autonomous sequential work on L0–L6 and assistant-managed merges into main after all required CI checks for the exact PR head, applicable actual visual review, and documented acceptance criteria pass. Do not wait for routine user approval or implement the whole roadmap in one task. Never bypass branch protection, disable tests, force-push, or merge unverified code. Stop for material blockers, scope changes, or local SDK gate N0.
- No Sandworm, no required SDK for offline stages, no AI training/fine-tuning/manual tuning/runtime inference. No CUDA/Blender dependency in base compiler. No paid GPU or self-hosted setup without user approval.
- Reuse-first: before implementing a substantial subsystem, inspect relevant upstream source, record pinned revision, license, dependencies, API and gaps. Prefer existing library or isolated reusable code; compare on PCG fixtures. Write a new algorithm only with documented reason. Preserve notices and separate vendor code from canonical PCG contracts. Prefer existing OSS after source/license audit; never publish SDK, credentials, unlicensed imagery, large generated assets, or model weights. No live mutable external data in fast tests. Pin dependencies.
- Keep canonical GIS observations, semantic geometry, architecture plans, render meshes and native adapters separate. Preserve footprint/holes and provenance; explicit rejected/fallback reasons. Stable seeds/IDs, meters, double world origins and float local vertices, explicit vertical datum.
- Include fixture/test, metrics and actual geometry-derived image for visual changes. Save generated artifacts outside tracked source; upload CI artifacts even when failure occurs. No screenshot mockup or claims of SDK/GPU test without actual run. Separate local vs Actions run and synthetic vs real data results.
- No change to goldens without explicit, documented visual review; assistant review is sufficient within agreed scope. No silent regression/disabled tests or relaxed thresholds just to pass. Record failing fixture and known gaps. Keep logs concise and reproducible.
- Before handing off: summarize changed files, commands, exact test status, CI run/artifact references, blockers, decisions and next single task. Do not report an unrun test as PASS. Local SDK G0–G12 are separate from CPU Lab.

## Consolidated numbered research-stage delivery

For the ongoing research PR, each numbered stage is one bounded implementation,
not a sequence of lettered micro-stages. Design and implement the entire stage,
run the relevant local tests, and prepare one coherent commit; preserve all
existing checks. Trigger CI for that exact head. Poll in the same working
session while tools permit, rather than handing off routine CI waiting to the
user. Download the actual CI artifact, verify its files, and visually inspect
the actual rendered output, including the previous stage when available.
Record metrics, CI links, known defects, decisions and the acceptance outcome
in the PR. Mark a stage accepted only after successful CI and actual visual
review; if tests, artifact access or the session are blocked, report the exact
blocker and NOT ACCEPTED. Do not invent a successful run or visual inspection.
Do not start the next stage or merge research PR #23 into main merely because
CI passed. No force pushes, disabled checks, or micro-commits to evade the gate.
