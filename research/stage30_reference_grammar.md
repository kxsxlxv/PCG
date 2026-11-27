# Stage 30 — reference-informed section grammar

## Private source research (offline only)

The two user-supplied sources were re-read on 2026-11-27. SHA256:
- Low-poly archive: 8082c844fd66ddcc1cd8606c55ab1dc74e02edc3977c662ed87f5beeeffbeb8e.
- Detailed glTF: 85bd5d73832d0cc3e2f6f9cd429b32c05614ba4ab0ee549435c43950c6b7d7f0.

The independent offline audit loaded node transforms and primitive indices,
triangulated strips, excluded degenerate triangles in transformed coordinates
and produced CPU depth-buffer views. Low-poly: 54 objects including alternate
color variants and ancillary structures, 14,916 non-degenerate triangles,
median 226, maximum 1,616. Detailed: 47 object groups, 50 mesh definitions,
670,734 non-degenerate triangles, median 8,886, maximum 51,898. Counts refer to
object groups, not a catalog of unique residential buildings. Raw units of the
low-poly set are not verified meters. The high-detail glTF has no material map
bindings, so its geometry is reviewed in clay; do not claim a textured source
comparison. No raw meshes, textures, private renders, filenames of individual
donor models or licensed atlas content are published here.

Observed useful patterns: section-based entries and stair cores, repeated roof
blocks, narrow and wide projecting window stacks, blank end walls, coherent
paint zones on lower and upper stories, framed office facades, clerestories and
service doors. Vertical vertex-level periodicity was examined using unique
height levels in 0.025 source-unit bins and autocorrelation over 2.4–3.75
source units; selected residential groups showed peaks around 2.90–3.10.
These are geometric periodicity hypotheses, not certified floor dimensions or
identifications of historical building series. The source library demonstrates
that silhouette and atlas detail can coexist at very low triangle counts.
It does not prove pixel-level reconstruction or appearance parity.

## Implementation and reuse

Reuse existing PCG archetypes, aperture and massing grammar, stage-26
placement/clearance/budget checks, compact GLB and alpha-cutout export, PCGF v1
ABI decoder, stage-27/28 material metadata and software EGL renderer and
stage-29 private overlay. New optional typed rules in pcg.reference_grammar;
8 generic families, explicitly authored dimensions, stable IDs, same canonical
shell. This is a new bounded rule set, not a new mesh simplifier or third-party
dependency. The core grammar is opt-in. Extra entries reserve blank ground
floor slots; full balcony stacks are selected as a unit with a conservative
30-triangle-per-anchor budget and 650-triangle reserve. Protruding front
quads and side closure cost 10 triangles per complete vertical bay (wide
two-bay version for tall tower), with exact host metric UV/slot aliases and
no canonical shell deletion. Roof cores and entry modules reuse existing
placement guards. Paint zoning affects opaque surfaces only and uses one
deterministic coherent RGB palette with existing material layers. No manual
per-building tuning or training. ModernGL imports moved inside the existing
render methods only; the new lane uses actual ctypes EGL, no stub renderer.

Private PBR optional, forbidden in CI. Public synthetic inputs cover two
cases per rule: rectangular, rotated, concave, courtyard and party-wall
variants. Both compared alternatives use the same stage-26 detail planner and
stage-29 PBR renderer. This is NOT a complete migration of stage-28/29
equipment refinements onto the new fixtures. Additional projected fronts
and entries use explicit research sidecars (not yet a native PCGF extension).
The exported GLBs, per-building placement manifests, alias and cutout records,
and GPU metadata including the exact cutout face order are written. No existing
test or threshold removed. The canonical shell, floor levels, footprint/holes,
resolved eave/top and roof type are identical on the matched inputs; visual
protrusions/roof equipment are separate attachment geometry.

## Local verification before publication

Both original shared PBR and private real PBR runs passed, each 48 actual
fixed-camera renders, 16 buildings, 8 rules. Same geometry and totals:
legacy Near/Lite 18,968/16,020; new Near/Lite 15,416/11,352;
maximum new Near 1,660 (cap 1,800). New 71 projection stacks, 37 supplemental
entries, 34 roof cores. Minimum whole-building Near/Lite silhouette IoU
0.9929178533. Four matched scenes (both baselines and candidates) checked
116,027 placement conditions; candidates 49,748. Candidate GPU slot oracle
80,976 samples, max paired edge drift 3.36e-6 m; C++ decoded 112-byte records.
1,728 actual EGL texture samples and 36 shared material layers. Both output
manifests 182/182 and input Stage-28 manifest 144/144 verified. Canonical host
vertices, records and cells invariant under reversed input order; rule plans
repeated independently. Visual QA added stronger base/vertical paint and
wider tall-tower columns, and changed party-wall camera to the visible side
for both matched renders. Local material colors and geometric structure are
clearer, but donor facade richness/individual atlas detail is not reproduced.
Near totals reduce partly because whole balcony stacks are selected more
conservatively, not an apples-to-apples geometry-only simplification.

Reproduce from repository root after preceding artifact stages exist:

    g++ -std=c++20 -O2 -Wall -Wextra -Werror lab/zero_tile_abi_check.cpp -o /tmp/pcg_zero_tile_abi
    PYTHONPATH=src LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless python lab/zero_reference_grammar.py /tmp/pcg_zero_tile_abi --out artifacts/zero-reference-grammar

Optional offline actual PBR: add --private-overlay /path/to/stage29/private/output
and use an output directory outside public CI. The private overlay maps remain
separate. No native UNIGINE or measured GPU performance claim, no verified
historical series catalog, no actual donor footprint reconstruction, and no
Far LOD or city-scale validation in this stage. Future work: calibrated
region/series selection, richer masonry/office/industrial families, structured
window/loggia variants, more diverse massing within GIS constraints, native
sidecar adapter and stage-29 equipment integration across new rules.
