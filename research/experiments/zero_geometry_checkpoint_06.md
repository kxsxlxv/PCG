# Checkpoint 06: executed shader reference

Head `a59c4a768db1850dac9ee9017c204578bd4b1db5`, [successful CI 45715190477](https://github.com/kxsxlxv/PCG/actions/runs/45715190477).
First attempt failed at EGL context creation (missing libEGL.so); the retry installed Mesa EGL and completed all 49 core tests and all workflow steps.

Actual GLSL compiled and executed using llvmpipe (OpenGL 4.5). Nine buildings, 210 triangles, one indexed draw per view in the lab, 13 material layers, six shared window/door modules. 17,275 visible wall lookup samples matched CPU opening classification; no boundary ambiguities. Material layer lookup exact; emission restricted to glazing; depth identical within each camera across modes. Dynamic illustrative probability 0.10/0.68 produced total sampled linear emission 255.698/1588.342.

The actual contact sheet was inspected: the silhouette and windows render correctly, but materials remain synthetic and the architecture is visually simplistic. No native UUSL compilation, hardware GPU timing, shadow/GI parity, measured flight anti-aliasing, or calibrated facade materials. No quality claim against Sandworm. Artifact 15069253286 contains zero-glsl/contact.png, report.json and module PNGs.

Next bounded increment: bridge existing approved synthetic GIS/DTM pilot to the neutral ABI while preserving complete canonical shell, facade plan, anchor and source provenance; keep unresolved items excluded.
