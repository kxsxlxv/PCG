"""L0 reproduction of historical experiments; not production compiler."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
report = {"commit": os.environ.get("GITHUB_SHA", "local"), "stages": []}
py = sys.executable

def run(name, args, verify):
    start = time.monotonic()
    log = OUT / (name + ".log")
    stage = {"name": name, "status": "FAIL"}
    try:
        with log.open("w", encoding="utf-8") as f:
            subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=240)
        stage["metrics"] = verify()
        stage["status"] = "PASS"
    except Exception as e:
        stage["error"] = repr(e)
    finally:
        stage["seconds"] = round(time.monotonic()-start, 3)
        report["stages"].append(stage)
        (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(name, stage["status"], stage.get("error", ""), flush=True)

def read(path):
    return json.loads((OUT/path).read_text())

def geometry():
    x=read("geometry.json")
    m=x["mesh_summary"]
    assert m["cases"]==20 and m["watertight"]==20 and m["consistent_winding"]==20 and m["positive_volume"]==20
    assert x["random_triangulation"]["cdt"]["nonzero_missing_cases"]==0
    assert (OUT/"geometry.glb").stat().st_size>0
    return m

def architecture():
    x=read("architecture/architecture_probe_results.json")
    assert x["cases"]==1200 and x["violations"]==0 and x["repeat_order_mismatches"]==0
    assert len(x["samples"])==18
    from PIL import Image
    with Image.open(OUT/"architecture/architecture_probe_contact.png") as im:
        assert im.width>100 and im.height>100
        assert any(lo!=hi for lo,hi in im.convert("RGB").getextrema())
    return x["totals"]

def materials():
    x=read("materials/results.json")
    assert x["maps_written"]==32 and x["facade"]["windows"]==48
    assert x["facade"]["mip_mean_error"]<1e-12
    return {"maps":32,"lit":x["facade"]["lit"]}

def cv():
    x=read("roof_color.json")
    assert x["accepted"]==600 and x["high_uncertainty_abstentions"]==600
    assert x["robust_mean_delta_e76"]<x["naive_mean_delta_e76"]
    return {"mean_delta_e":x["robust_mean_delta_e76"]}

def streaming():
    x=read("streaming.json")
    assert len(x["tests"])==12 and all(t["peak_fine_payload_mib"]<=96 for t in x["tests"])
    return {"scenarios":len(x["tests"])}

def coords():
    x=read("coordinates.json")
    assert x["samples"]==1681 and x["tile_local_float_max_error_m"]<.0001
    return {"tile_local_error_m":x["tile_local_float_max_error_m"]}

def assets():
    x=read("assets/test_results.json")
    assert len(x["tests"])==40 and len(x["default_triangles"])==3
    return {"recipes":len(x["tests"]),"triangles":x["default_triangles"]}

run("geometry",[py,"research/03_04_geometry/geometry_probe.py","--output",str(OUT/"geometry.json"),"--models",str(OUT/"geometry.glb")],geometry)
run("architecture",[py,"research/03_04_geometry/architecture_probe.py","--out",str(OUT/"architecture")],architecture)
run("assets",["node","research/05_asset_factory/asset_probe.mjs",str(OUT/"assets")],assets)
run("materials",[py,"research/06_materials/material_probe.py","--output",str(OUT/"materials"),"--size","512"],materials)
run("cv",[py,"research/07_cv/roof_color_probe.py","--output",str(OUT/"roof_color.json")],cv)
run("streaming",["node","research/08_rendering/streaming_probe.mjs",str(OUT/"streaming.json")],streaming)
run("coordinates",[py,"research/09_integration/coordinate_probe.py",str(OUT/"coordinates.json")],coords)
sys.exit(0 if all(s["status"]=="PASS" for s in report["stages"]) else 1)
