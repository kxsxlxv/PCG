"""Export an original marker and existing procedural PCG fixtures for the SDK lab."""

import argparse
from hashlib import file_digest
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab"))

from pcg.footprint import Footprint
from pcg.materials import MATERIALS, texture_maps
from pcg.occupancy import EMISSION
from pcg.terrain import footprint_sample_points
from render_material_uv import export_reload, image
from render_neighborhood import build_block


def read_plaster_pack(path):
    """Read one ambientCG Painted Plaster 015 2K-PNG archive for the local A/B."""
    prefix = "PaintedPlaster015_2K-PNG_"
    maps = {}
    with ZipFile(path) as archive:
        for name, channel in (("Color", "RGB"), ("NormalGL", "RGB"), ("Roughness", "L"),
                              ("AmbientOcclusion", "L")):
            member = archive.getinfo(prefix + name + ".png")
            if member.file_size > 40 * 1024**2:
                raise ValueError(f"Plaster map exceeds preview budget: {member.filename}")
            with Image.open(BytesIO(archive.read(member))) as image:
                if image.size != (2048, 2048):
                    raise ValueError(f"Unexpected plaster map size: {member.filename}")
                maps[name] = image.convert(channel)
    roughness = np.asarray(maps["Roughness"])
    orm = Image.fromarray(np.stack((np.asarray(maps["AmbientOcclusion"]), roughness,
                                   np.zeros_like(roughness)), axis=-1))
    return {"basecolor": maps["Color"], "normal": maps["NormalGL"], "orm": orm}


def write_sdk_maps(output, maps):
    """Translate glTF texture semantics to the SDK mesh_base texture layout."""
    for key, images in maps.items():
        directory = output / "maps" / key
        directory.mkdir(parents=True)
        images["basecolor"].save(directory / "albedo.png")
        normal = np.asarray(images["normal"]).copy()
        normal[..., 1] = 255 - normal[..., 1]
        Image.fromarray(normal).save(directory / "normal.png")
        orm = np.asarray(images["orm"])
        Image.fromarray(orm[..., 0]).save(directory / "ao.png")
        specular = np.full_like(orm[..., 0], 128)
        Image.fromarray(np.stack((orm[..., 2], orm[..., 1], specular), axis=-1)).save(
            directory / "shading.png"
        )


def write_sdk_mesh(path, meshes):
    """Write geometry from the already verified GLB round-trip in canonical Z-up."""
    with path.open("w", encoding="ascii") as file:
        print("PCG_PREVIEW_V1", len(meshes), file=file)
        for mesh in sorted(meshes, key=lambda m: m.visual.material.name):
            key = mesh.visual.material.name
            _, _, variant = key.partition("__")
            print(key, len(mesh.faces), *EMISSION[variant or "off"], file=file)
            indices = mesh.faces.flatten()
            vertices = np.column_stack((mesh.vertices[indices], mesh.vertex_normals[indices],
                                        mesh.visual.uv[indices]))
            np.savetxt(file, vertices, fmt="%.9g")


def write_expected(output, records):
    with (output / "expected.txt").open("w", encoding="ascii") as file:
        for name, record in records.items():
            extent = " ".join(f"{n:.8f}" for point in record["bounds"] for n in point)
            print(name, record["triangles"], len(record["materials"]), extent, file=file)


def write_layout(output, bounds):
    with (output / "layout.txt").open("w", encoding="ascii") as file:
        print(*(coordinate for point in bounds for coordinate in point[:2]), file=file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", choices=("profiles", "neighborhood"), default="profiles")
    parser.add_argument("--input-fixture", type=Path, default=ROOT / "tests/fixtures/l4_neighborhood.json")
    parser.add_argument("--profile", choices=("house", "apartments", "office"))
    parser.add_argument("--seed", type=int, default=17, help="Deterministic occupancy seed")
    parser.add_argument("--lod", type=int, choices=(0, 1), default=0)
    parser.add_argument("--plaster-pack", type=Path, help="Optional local PaintedPlaster015 2K-PNG archive")
    args = parser.parse_args()

    fixture = json.loads(args.input_fixture.read_text(encoding="utf-8"))
    fixture["seed"] = args.seed
    if args.fixture == "profiles":
        first = {}
        for building in fixture["buildings"]:
            first.setdefault(building["profile"], building)
        fixture["buildings"] = list(first.values())
    if args.profile:
        fixture["buildings"] = [c for c in fixture["buildings"] if c["profile"] == args.profile]
    if not fixture["buildings"] or len(fixture["buildings"]) > 16:
        parser.error("The preview requires 1 to 16 buildings")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", case["id"]) for case in fixture["buildings"]):
        parser.error("Preview building IDs must be safe file names")

    day, night, buildings, _ = build_block(fixture, args.lod)
    marker = {
        "ceramic_roof": [((1.0, 2.0, 3.0), (5.0, 2.0, 3.0), (1.0, 4.5, 3.0))],
        "glass_proxy": [((1.0, 2.0, 3.0), (1.0, 2.0, 4.25), (5.0, 2.0, 3.0))],
    }
    maps = {
        key: {name: image(values) for name, values in texture_maps(MATERIALS[key]).items()}
        for key in sorted(day.keys() | marker.keys())
    }
    args.output.mkdir(parents=True, exist_ok=False)
    native_maps = maps
    comparison = None
    if args.plaster_pack:
        if "plaster_cream" not in day:
            parser.error("The plaster comparison needs a house in the fixture")
        native_maps = {**maps, "plaster_cream": read_plaster_pack(args.plaster_pack)}
        with args.plaster_pack.open("rb") as file:
            checksum = file_digest(file, "sha256").hexdigest()
        comparison = {"role": "plaster_cream", "archive": args.plaster_pack.name, "sha256": checksum,
                      "source": "https://ambientcg.com/view?id=PaintedPlaster015", "license": "CC0-1.0",
                      "provisional_repeat_m": MATERIALS["plaster_cream"].repeat_m,
                      "scope": "native preview texture override"}
    write_sdk_maps(args.output, native_maps)
    records = {}
    def export(name, groups, at_night=False):
        path = args.output / f"{name}.glb"
        meshes = export_reload(groups, maps, path, night=at_night)
        write_sdk_mesh(args.output / f"{name}.pcg", meshes)
        vertices = np.vstack([m.vertices for m in meshes])
        records[name] = {
            "triangles": sum(len(m.faces) for m in meshes),
            "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
            "materials": sorted(groups),
        }

    export("marker", marker)
    export("day", day)
    export("night", night, at_night=True)
    with (args.output / "placement.txt").open("w", encoding="ascii") as placement:
        for case in sorted(fixture["buildings"], key=lambda value: value["id"]):
            key = case["id"]
            polygon = Footprint(key, case["outer"], case.get("holes", [])).polygon()
            samples = footprint_sample_points(polygon, 2.0)
            if samples is None:
                raise ValueError(f"Sample budget exceeded for {key}")
            x, y, _, _ = polygon.bounds
            print(key, x, y, len(samples), file=placement)
            print("\n".join(f"{px - x:.12g} {py - y:.12g}" for px, py in samples), file=placement)
            local = dict(case, outer=[(px - x, py - y) for px, py in case["outer"]],
                         holes=[[(px - x, py - y) for px, py in ring] for ring in case.get("holes", [])])
            local_day, local_night, _, _ = build_block({**fixture, "buildings": [local]}, args.lod)
            export(f"day-{key}", local_day)
            export(f"night-{key}", local_night, at_night=True)
    (args.output / "expected.json").write_text(
        json.dumps({"fixture": args.fixture, "seed": args.seed, "lod": args.lod, "buildings": buildings,
                    "comparison": comparison,
                    "exports": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_expected(args.output, records)
    write_layout(args.output, records["day"]["bounds"])


if __name__ == "__main__":
    main()
