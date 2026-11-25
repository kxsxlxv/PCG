"""Use Blender headlessly to decode geometry in selected compressed reference GLBs."""

from pathlib import Path
import sys

import bpy


arguments = sys.argv[sys.argv.index("--") + 1:]
if len(arguments) < 2:
    raise ValueError("Pass an output directory and at least one GLB after --")
output = Path(arguments[0]).resolve()
output.mkdir(parents=True, exist_ok=False)
for argument in arguments[1:]:
    source = Path(argument).resolve(strict=True)
    destination = output / source.name
    if destination.exists() or source.suffix.lower() != ".glb":
        raise ValueError(f"Expected distinct GLB inputs: {source}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(source)):
        raise RuntimeError(f"Blender could not import {source}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"No reference geometry in {source}")
    for mesh in meshes:
        mesh.data.calc_loop_triangles()
    count = sum(len(mesh.data.loop_triangles) for mesh in meshes)
    print(f"{source.name}: {len(meshes)} meshes, {count} triangles", flush=True)
    if "FINISHED" not in bpy.ops.export_scene.gltf(
        filepath=str(destination), export_format="GLB", export_image_format="NONE"
    ):
        raise RuntimeError(f"Blender could not export {destination}")
