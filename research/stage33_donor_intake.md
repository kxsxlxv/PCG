# Stage 33 candidate — private donor reverse-engineering intake

This is an offline intake tool, not completed reconstruction. No donor assets
or donor-derived images belong in the public repository, public CI logs or
artifacts. A successful synthetic test does not constitute donor acceptance.

Place both original files and the three corresponding maps in one private
directory. Run from the repository root with its Python dependencies:
```bash
PYTHONPATH=src:lab python lab/reference_intake33.py --low /private/buildings_2012.zip --high /private/detailed.gltf --out /private/stage33-catalog
```
The output folder must not already exist. The catalog contains per-group
transformed triangle counts, bounds, areas by orientation, UV coverage and
relative density, material bindings and image decoding status, approximate
ground outline and period hypotheses. It preserves source names locally.
No automatic assignment of unbound atlas maps. glTF is converted from Y-up;
OBJ axes and source scale require independent validation. No exact historic
series recognition or inference of metric dimensions from raw units.

Next acceptance requires an actual catalog for both packs, independent metric
calibration, original textured views with verified material bindings, semantic
module measurements, original-versus-procedural matched images and a budget
comparison on identical objects. No changes to Stage 32 production behavior.

The optional local roughness audit segments dark, nearly rectangular regions
at 8x8 pixel block resolution and writes a private diagnostic PNG. It does
not treat them as unique window models or authorize source-map bindings.
