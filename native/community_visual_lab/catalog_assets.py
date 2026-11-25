"""Inspect a small selection of local asset files without importing them."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import struct
from zipfile import BadZipFile, ZipFile

from PIL import Image


def inspect(path):
    with path.open("rb") as stream:
        checksum = file_digest(stream, "sha256").hexdigest()
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": checksum,
              "usage": "local reference, rights unverified"}
    if path.suffix == ".glb":
        with path.open("rb") as stream:
            magic, version, _, size, chunk = struct.unpack("<4sIII4s", stream.read(20))
            if (magic, version, chunk) != (b"glTF", 2, b"JSON") or size > 8 * 1024**2:
                raise ValueError("unsupported GLB JSON header")
            gltf = json.loads(stream.read(size))
        accessors = gltf["accessors"]
        primitives = [p for mesh in gltf["meshes"] for p in mesh["primitives"]]
        record.update(
            generator=gltf["asset"].get("generator"),
            embedded_copyright=gltf["asset"].get("copyright"),
            mesh_primitives=len(primitives),
            indexed_or_sequential_triangles=sum(
                accessors[p.get("indices", p["attributes"]["POSITION"])]["count"] // 3
                for p in primitives if p.get("mode", 4) == 4
            ),
            position_accessor_bounds_y_up=[
                [accessors[p["attributes"]["POSITION"]].get(key) for key in ("min", "max")]
                for p in primitives
            ],
            geometry_extensions=gltf.get("extensionsRequired", []),
            material_names=[m.get("name") for m in gltf.get("materials", [])],
            adjacent_lods=sorted(p.name for p in path.parent.glob(path.name.replace("lod0", "lod?"))),
        )
    elif path.suffix == ".zip":
        with ZipFile(path) as archive:
            record["archive_members"] = sorted(archive.namelist())[:40]
    elif path.suffix == ".jpg":
        with Image.open(path) as image:
            image.load()
            record.update(image_size=image.size, image_mode=image.mode)
    return record


def summarize_root(root):
    try:
        children = sorted(root.iterdir())
        return {"path": str(root), "directories": [p.name for p in children if p.is_dir()][:30],
                "file_extensions": dict(Counter(p.suffix.lower() for p in children if p.is_file()))}
    except OSError as error:
        return {"path": str(root), "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk_project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, default=Path("D:/Graphics"))
    args = parser.parse_args()
    project = args.sdk_project.resolve()
    pbr = args.library / "Materials_Library/PBR Materials"
    roots = [project / name for name in ("buildings", "props", "materials", "asset_sources", "source/third_party")]
    roots.extend([args.library / "Assets", pbr / "surface", pbr / "Leaking"])
    selected = [project / name for name in (
        "buildings/apartment-block-v2-lod0.glb", "buildings/office-tower-v2-lod0.glb",
        "buildings/concrete-frame-build-v1-lod0.glb", "props/ac-unit-on-wall-bracket-v1-lod0.glb",
        "props/rooftop-air-handler-v1-lod0.glb", "props/rooftop-vent-stacks-v1-lod0.glb",
        "materials/PaintedPlaster015_2K-PNG.zip", "materials/Concrete034_2K-PNG.zip",
        "materials/Bricks060_2K-PNG.zip", "materials/RoofingTiles012A_2K-PNG.zip",
        "materials/corrugated_iron_2k.zip", "materials/bitumen_2k.zip",
        "materials/Granite002A_2K-PNG.zip", "materials/PaintedMetal010_2K-PNG.zip",
        "materials/PaintedWood001_2K-PNG.zip", "materials/kloppenheim_06_puresky_4k.exr",
        "asset_sources/references/Sochi_Adler/manifest.json",
        "asset_sources/references/Sochi_Adler/01_private_lowrise/R001.jpg",
    )]
    selected.extend(pbr / "Leaking" / path for path in (
        "Leaking011C_4K-JPG.zip", "Leaking009_2K-JPG/Leaking009_2K-JPG_Opacity.jpg",
    ))
    for category in ("brick", "concrete", "plaster", "roof", "metal"):
        root = pbr / "surface" / category
        roots.append(root)
        try:
            archives = sorted(root.glob("*.zip"))[:10]
            selected.extend(archives[:1])
        except OSError:
            continue
    records = []
    for path in selected:
        try:
            if path.is_symlink() or path.stat().st_size > 250_000_000:
                raise ValueError("external link or file above 250 MB inspection budget")
            records.append(inspect(path))
        except (OSError, ValueError, KeyError, IndexError, struct.error, BadZipFile) as error:
            records.append({"path": str(path), "error": str(error)})
    catalog = {"roots": [summarize_root(root) for root in roots], "selected": records}
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
