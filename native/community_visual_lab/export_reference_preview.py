"""Stage decoded reference GLBs as neutral geometry for a separate local gallery."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import box
import trimesh

from export_preview import write_expected, write_layout, write_sdk_maps, write_sdk_mesh
from pcg.terrain import footprint_sample_points


def geometry(path):
    scene = trimesh.load(path, force="scene", process=False)
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    parts = []
    for name in scene.graph.nodes_geometry:
        transform, key = scene.graph[name]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(rotation @ transform)
        if mesh.visual.uv is None:
            raise ValueError(f"Reference primitive has no UV0: {path.name}")
        parts.append(mesh)
    if not parts:
        raise ValueError(f"Reference has no mesh primitives: {path.name}")
    vertices = np.concatenate([mesh.vertices for mesh in parts])
    triangles = np.concatenate([mesh.triangles for mesh in parts])
    uv = np.concatenate([mesh.visual.uv[mesh.faces] for mesh in parts]).reshape(-1, 2)
    normals = np.concatenate([mesh.vertex_normals[mesh.faces] for mesh in parts]).reshape(-1, 3)
    material = trimesh.visual.material.PBRMaterial(name="reference_matte")
    origin = vertices.min(0)
    triangle_vertices = triangles.reshape(-1, 3) - origin
    mesh = trimesh.Trimesh(vertices=triangle_vertices, faces=np.arange(len(triangle_vertices)).reshape(-1, 3),
                          vertex_normals=normals, process=False,
                          visual=trimesh.visual.texture.TextureVisuals(uv=uv, material=material))
    return mesh, origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("models", nargs="+", type=Path, help="Uncompressed Blender exports of selected donors")
    args = parser.parse_args()
    if len(set(p.stem for p in args.models)) != len(args.models) or len(args.models) > 16:
        parser.error("Expected up to 16 distinct model names")
    args.output.mkdir(parents=True, exist_ok=False)
    write_sdk_maps(args.output, {"reference_matte": {
        "basecolor": Image.new("RGB", (16, 16), (165, 170, 178)),
        "normal": Image.new("RGB", (16, 16), (128, 128, 255)),
        "orm": Image.new("RGB", (16, 16), (255, 192, 0)),
    }})
    records = {}
    gallery = []
    x = 0.0
    maximum_y = 0.0
    with (args.output / "placement.txt").open("w", encoding="ascii") as placement:
        for path in args.models:
            mesh, native_origin = geometry(path)
            low, high = mesh.bounds
            samples = footprint_sample_points(box(low[0], low[1], high[0], high[1]), 2.0)
            if samples is None:
                raise ValueError(f"Reference dimensions exceed preview budget: {path.name}")
            print(path.stem, x, 0, len(samples), file=placement)
            print("\n".join(f"{px:.12g} {py:.12g}" for px, py in samples), file=placement)
            name = f"day-{path.stem}"
            write_sdk_mesh(args.output / f"{name}.pcg", [mesh])
            records[name] = {"bounds": mesh.bounds.tolist(), "triangles": len(mesh.faces),
                             "materials": ["reference_matte"]}
            gallery.append({"file": path.name, "source_minimum_z_up": native_origin.tolist(),
                            "gallery_anchor_xy": [x, 0.0], "size_m": (high - low).tolist()})
            x += float(high[0]) + max(0.6, float(high[0]) * 0.15)
            maximum_y = max(maximum_y, float(high[1]))
    write_expected(args.output, records)
    write_layout(args.output, [(0.0, 0.0), (x, maximum_y)])
    (args.output / "gallery.json").write_text(
        json.dumps({"mode": "reference_geometry", "assets": gallery, "exports": records}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
