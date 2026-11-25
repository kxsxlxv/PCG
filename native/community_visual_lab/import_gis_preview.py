"""Stage verified L5 compiled buildings with one shared Cartesian preview origin."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import trimesh

from export_preview import write_expected, write_layout, write_sdk_maps, write_sdk_mesh
from pcg.footprint import Footprint
from pcg.terrain import footprint_sample_points


def read_glb(root, descriptor):
    relative = descriptor["path"].replace("\\", "/")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.stat().st_size != descriptor["bytes"]:
        raise ValueError("GLB path or size differs from the compiler report")
    data = path.read_bytes()
    if sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError("GLB checksum differs from the compiler report")
    scene = trimesh.load(BytesIO(data), file_type="glb", force="scene", process=False)
    geometry = []
    for name in scene.graph.nodes_geometry:
        transform, key = scene.graph[name]
        np.testing.assert_allclose(transform, np.eye(4), atol=1e-8, rtol=0)
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        geometry.append(mesh)
    if len(geometry) != len(scene.geometry) or sum(len(m.faces) for m in geometry) != descriptor["triangles"]:
        raise ValueError("Unexpected GLB instances or triangle count")
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled", type=Path, help="L5 directory containing report.json")
    parser.add_argument("output", type=Path, help="New native preview package directory")
    parser.add_argument("--shared-origin", type=float, nargs=2, metavar=("X", "Y"))
    args = parser.parse_args()
    root = args.compiled.resolve(strict=True)
    report = json.loads((root / "report.json").read_bytes())
    if report["schema_version"] != 1:
        raise ValueError("Unsupported L5 report schema")
    buildings = sorted((b for b in report["records"] if b["status"] == "compiled"),
                       key=lambda b: b["building_id"])
    if not 1 <= len(buildings) <= 16:
        raise ValueError("The preview requires 1 to 16 compiled buildings")
    ids = [b["building_id"] for b in buildings]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", key) for key in ids):
        raise ValueError("Expected distinct building IDs safe for preview file names")
    anchors = np.asarray([b["anchor_local_z_up"] for b in buildings], dtype=float)
    shared = anchors[:, :2].min(0) if args.shared_origin is None else np.asarray(args.shared_origin)
    if not np.isfinite(anchors).all() or not np.isfinite(shared).all():
        raise ValueError("Nonfinite preview origin or canonical anchor")
    args.output.mkdir(parents=True, exist_ok=False)
    textures = {}
    exports = {}
    preview_anchors = []
    layout = []
    with (args.output / "placement.txt").open("w", encoding="ascii") as placement:
        for record, anchor in zip(buildings, anchors, strict=True):
            key = record["building_id"]
            footprint = record["footprint"]
            polygon = Footprint(key, footprint["exterior"], footprint["holes"]).polygon()
            samples = footprint_sample_points(polygon, 2.0)
            if samples is None:
                raise ValueError(f"Sample budget exceeded for {key}")
            xy = anchor[:2] - shared
            print(key, *xy, len(samples), file=placement)
            print("\n".join(f"{x:.12g} {y:.12g}" for x, y in samples), file=placement)
            preview_anchors.append({"id": key, "preview_xy": xy.tolist(), "canonical_anchor": anchor.tolist(),
                                    "canonical_projected_anchor": record["anchor_projected_enh"]})
            for mode in ("day", "night"):
                name = f"{mode}-{key}"
                meshes = read_glb(root, record["exports"][mode])
                for mesh in meshes:
                    material = mesh.visual.material
                    base = material.name.partition("__")[0]
                    maps = {"basecolor": material.baseColorTexture.convert("RGB"),
                            "normal": material.normalTexture.convert("RGB"),
                            "orm": material.metallicRoughnessTexture.convert("RGB")}
                    if base in textures:
                        for channel, value in maps.items():
                            np.testing.assert_array_equal(np.asarray(value), np.asarray(textures[base][channel]))
                    else:
                        textures[base] = maps
                vertices = np.vstack([m.vertices for m in meshes])
                bounds = [vertices.min(0), vertices.max(0)]
                np.testing.assert_allclose(bounds, record["render_bounds_building_local_z_up"], atol=2e-6, rtol=0)
                exports[name] = {"triangles": sum(len(m.faces) for m in meshes),
                                 "bounds": [point.tolist() for point in bounds],
                                 "materials": sorted(m.visual.material.name for m in meshes)}
                write_sdk_mesh(args.output / f"{name}.pcg", meshes)
                if mode == "day":
                    layout.append(np.asarray(bounds)[:, :2] + xy)
    write_sdk_maps(args.output, textures)
    write_expected(args.output, exports)
    write_layout(args.output, [np.min([b[0] for b in layout], axis=0), np.max([b[1] for b in layout], axis=0)])
    provenance = {"world_frame": report["identity"]["frame"], "package_digest": report["package_digest"],
                  "shared_preview_origin_xy": shared.tolist(), "preview_anchors": preview_anchors,
                  "exports": exports}
    (args.output / "gis_preview.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
