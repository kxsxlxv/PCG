"""Execute pinned upstream OSM Building Grammar without Blender.

Diagnostic baseline only: does not replace PCG massing or claim true apertures.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from vendor.osm_building_grammar.grammar import generate_building_spec
from vendor.osm_building_grammar.config import BuildingGrammarConfig, FacadeStyleConfig
from render_l2_massing import raster, VIEWS, SIZE

REVISION = "81f1b50eae048645b2e620056a7cb2c84709bcfa"
PROFILES = ("house", "apartments", "office")
OUTER = [(0.,0.,0.),(12.,0.,0.),(12.,8.,0.),(0.,8.,0.)]


def baseline(profile):
    style = FacadeStyleConfig(name=profile)
    style.antenna.enabled = False
    style.ledge.enabled = profile != "house"
    style.balcony.enabled = profile == "apartments"
    style.wall_color = {"house":(.76,.66,.49,1), "apartments":(.78,.75,.68,1),
                        "office":(.55,.60,.63,1)}[profile]
    if profile == "office":
        style.window.width = 2.2
        style.window.spacing = 2.8
        style.window.height = 2.0
    levels = 2 if profile == "house" else 3
    height = 6. if levels == 2 else 9.
    config = BuildingGrammarConfig(styles=[style])
    config.roof.edge_enabled = False
    config.roof.overhang = 0.
    tags = {"building":profile, "building:levels":str(levels),
            "height":str(height), "roof:shape":"flat"}
    return generate_building_spec(OUTER,tags,config,source_name=profile),levels,height


def mesh_for(spec):
    verts=[]; faces=[]; colors=[]
    for s in spec.meshes:
        offset=len(verts)
        verts.extend(s.vertices)
        for f in s.faces:
            assert len(f)>=3 and min(f)>=0 and max(f)<len(s.vertices), s.name
            # Diagnostic conversion of upstream box/quad geometry for these
            # rectangular fixtures only; not a general n-gon triangulator.
            for j in range(1,len(f)-1):
                faces.append((offset+f[0],offset+f[j],offset+f[j+1]))
                colors.append(np.clip(np.array(s.color)*255,0,255).astype(np.uint8))
    m=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    assert np.isfinite(m.vertices).all()
    assert (m.area_faces>1e-10).all(), "degenerate donor triangle"
    m.unmerge_vertices()
    m.visual.vertex_colors=np.repeat(np.array(colors),3,axis=0)
    return m


def digest(spec):
    return hashlib.sha256(json.dumps(asdict(spec),sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def main():
    out=Path("artifacts/l3-oss-baseline")
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    sheet=Image.new("RGB",(4*SIZE,len(PROFILES)*SIZE),"white")
    for row,profile in enumerate(PROFILES):
        start=time.perf_counter()
        spec,levels,height=baseline(profile)
        duration=time.perf_counter()-start
        assert spec.levels==levels and abs(spec.height-height)<1e-9
        assert digest(spec)==digest(baseline(profile)[0])
        roles=Counter(s.role for s in spec.meshes)
        assert roles["facade"]>0 and roles["window"]>0 and roles["door"]>0
        m=mesh_for(spec)
        bounds=m.bounds.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
        path=out/(profile+".glb")
        path.write_bytes(m.export(file_type="glb"))
        loaded=trimesh.load(path,force="mesh",process=False)
        loaded.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        np.testing.assert_allclose(loaded.bounds,bounds,atol=2e-6)
        assert len(loaded.faces)==len(m.faces)
        for i,(label,view,up) in enumerate(VIEWS):
            image,_=raster(loaded,view,up)
            ImageDraw.Draw(image).text((10,10),profile+" / "+label,fill=(0,0,0))
            image.save(out/(profile+"_"+label+".png"))
            sheet.paste(image,(i*SIZE,row*SIZE))
        records.append({"profile":profile,"levels":levels,"wall_height_m":height,
            "mesh_count":len(spec.meshes),"triangles":len(m.faces),"roles":dict(roles),
            "digest":digest(spec),"generation_ms":duration*1000,"bounds":bounds.tolist(),
            "glb_bytes":path.stat().st_size})
    sheet.save(out/"all_contact.png")
    report={"upstream_revision":REVISION,"profiles":records,"blender_required":False,
            "limitations":["rectangular diagnostic fixtures only","overlay windows, not apertures",
                           "no common LOD plan or party-wall validation","not production integration"]}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
