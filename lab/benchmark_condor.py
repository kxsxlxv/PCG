"""Pinned Condor generator research lane. Synthetic inputs; no production changes."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from render_l2_massing import raster
from shapely.geometry import Polygon as SPolygon
from shapely.ops import unary_union

REVISION = "26a5d22481c03b0a491abfa498a5ab795cb223b7"
OUT = Path("artifacts/condor")
OUT.mkdir(parents=True, exist_ok=True)
donor = Path(os.environ.get("CONDOR_SOURCE", "/tmp/condor-pinned"))
if not donor.exists():
    subprocess.run(["git", "init", str(donor)], check=True)
    subprocess.run(["git", "-C", str(donor), "fetch", "--depth=1",
                    "https://github.com/jlgabriel/condor-buildings-generator.git", REVISION], check=True)
    subprocess.run(["git", "-C", str(donor), "checkout", "--detach", "FETCH_HEAD"], check=True)
assert subprocess.check_output(["git","-C",str(donor),"rev-parse","HEAD"],text=True).strip()==REVISION
sys.path.insert(0,str(donor.resolve()))

from condor_buildings.models.building import BuildingRecord, BuildingCategory, RoofType
from condor_buildings.models.geometry import Polygon, Point2D
from condor_buildings.generators.building_generator import generate_building_separated, POLYSKEL_AVAILABLE
from condor_buildings.processing.mesh_grouper import MeshGrouper
from condor_buildings.io.obj_exporter import export_obj

def rect(w,d): return [(0,0),(w,0),(w,d),(0,d)]

CASES=[
    ("gable_house","house","gabled",2,6,rect(12,8),[]),
    ("hip_house","house","hipped",1,3,rect(10,8),[]),
    ("L_house","house","hipped",2,6,[(0,0),(12,0),(12,5),(7,5),(7,10),(0,10)],[]),
    ("apartment","apartment","flat",5,15,rect(30,12),[]),
    ("courtyard","apartment","flat",4,12,rect(30,24),[[(8,6),(8,18),(22,18),(22,6)]]),
    ("warehouse","industrial","flat",1,6,rect(38,22),[]),
]
def record(case, offset=(0,0), seed=42):
    name,cat,roof,floors,h,outer,holes=case
    def pts(r):return [Point2D(x+offset[0],y+offset[1]) for x,y in r]
    return BuildingRecord(osm_id=name,category=BuildingCategory(cat),footprint=Polygon(pts(outer),[pts(r) for r in holes]),floors=floors,height_m=h,roof_type=RoofType(roof),seed=seed)

def run(case,lod,offset=(0,0),seed=42):
    b=record(case,offset,seed)
    start=time.perf_counter()
    r=generate_building_separated(b,overhang=0.5 if lod==0 else 0.)
    elapsed=(time.perf_counter()-start)*1000
    return b,r,elapsed

def mesh_for(r):
    verts=[]; faces=[]; colors=[]
    for part,color in [(r.walls,(218,205,180,255)),(r.roof,(150,75,55,255))]:
        offset=len(verts); verts.extend(part.vertices)
        assert len(part.faces)==len(part.face_uvs), "UV face mismatch"
        for f,uv in zip(part.faces,part.face_uvs):
            assert len(f)==len(uv) and min(f)>=1 and max(f)<=len(part.vertices)
            assert min(uv)>=1 and max(uv)<=len(part.uvs)
            for k in range(1,len(f)-1):
                faces.append(tuple(offset+f[j]-1 for j in (0,k,k+1)));colors.append(color)
    m=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    assert np.isfinite(m.vertices).all() and len(m.faces)>0
    assert np.isfinite(np.asarray(r.walls.uvs)).all() and np.isfinite(np.asarray(r.roof.uvs)).all()
    degenerate=int(np.sum(m.area_faces<1e-10))
    m.unmerge_vertices()
    m.visual.vertex_colors=np.repeat(np.asarray(colors,dtype=np.uint8),3,axis=0)
    return m,degenerate

def reload_glb(m,path):
    m=m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
    path.write_bytes(m.export(file_type="glb"))
    m=trimesh.load(path,force="mesh",process=False)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    return m

def draw(m,view,center,ppm,size):
    image,depth=raster(m,view,[0,1,0] if view==[0,0,1] else [0,0,1],center=center,pixels_per_m=ppm,size=size)
    assert np.isfinite(depth).sum()>100
    return image

def roof_coverage(case,r):
    target=SPolygon(case[5],case[6])
    polys=[]
    for f in r.roof.faces:
        tri=SPolygon([r.roof.vertices[i-1][:2] for i in f])
        if tri.is_valid and tri.area>1e-9: polys.append(tri)
    actual=unary_union(polys)
    return {"target_area":target.area,"projected_roof_area":actual.area,
            "intersection_area":target.intersection(actual).area,
            "footprint_iou":target.intersection(actual).area/target.union(actual).area,
            "hole_covered_area":sum(actual.intersection(SPolygon(h)).area for h in case[6])}

def main():
    report={"revision":REVISION,"synthetic":True,"original_textures_present":False,
            "renderer":"PCG CPU z-buffer, role colors (no PBR or native textures)",
            "polyskel_available":bool(POLYSKEL_AVAILABLE),"cases":[]}
    size=300
    sheet=Image.new("RGB",(size*4,size*len(CASES)+30),(245,243,238))
    ImageDraw.Draw(sheet).text((8,8),"CONDOR / actual donor meshes / synthetic footprints / role colors only",fill=(0,0,0))
    for i,case in enumerate(CASES):
        name=case[0]
        for lod in (0,1):
            b,r,ms=run(case,lod)
            m,deg=mesh_for(r)
            export_obj([r.walls,r.roof],str(OUT/f"{name}_lod{lod}.obj"))
            loaded=reload_glb(m,OUT/f"{name}_lod{lod}.glb")
            np.testing.assert_allclose(loaded.bounds,m.bounds,atol=1e-5)
            assert len(loaded.faces)==len(m.faces)
            # Fixed scale across LODs for a given case; include roof overhang.
            outer=np.asarray(case[5]); low=outer.min(axis=0);high=outer.max(axis=0)
            center=(*((low+high)/2),case[4]/2+1.5)
            ppm=(size-46)/(max(high-low)+case[4]*0.5+4)
            for j,view in enumerate(([1,-1,0.8],[0,0,1])):
                im=draw(loaded,view,center,ppm,size)
                ImageDraw.Draw(im).text((8,8),f"{name} LOD{lod} {'iso' if j==0 else 'top'} / {len(m.faces)} tris",fill=(0,0,0))
                im.save(OUT/f"{name}_lod{lod}_{j}.png")
                sheet.paste(im,((lod*2+j)*size,i*size+30))
            report["cases"].append({"name":name,"lod":lod,"requested":case[2],"actual":r.actual_roof_type.value,
                "fallback":r.fallback_reason,"warnings":r.warnings,"triangles":len(m.faces),
                "degenerate_triangles":deg,"source_vertices":len(r.walls.vertices)+len(r.roof.vertices),
                "generation_ms":ms,"bounds":m.bounds.tolist(),"roof_coverage":roof_coverage(case,r),"building_warnings":b.warnings})
    sheet.save(OUT/"contact.png")
    # 64 buildings, same synthetic fixture mix; no simulation of GPU timing.
    for lod in (0,1):
        meshes=[];grouper=MeshGrouper();start=time.perf_counter()
        for i in range(64):
            c=CASES[i%len(CASES)]
            b,r,_=run(c,lod,((i%8)*48,(i//8)*48),seed=42+i)
            grouper.add_building(b,r)
            meshes.append(mesh_for(r)[0])
        total_ms=(time.perf_counter()-start)*1000
        m=trimesh.util.concatenate(meshes)
        loaded=reload_glb(m,OUT/f"block_lod{lod}.glb")
        im=draw(loaded,[1,-1,1.4],(185,185,0),1.25,720)
        ImageDraw.Draw(im).text((10,10),f"64 synthetic buildings / LOD{lod} / {len(m.faces)} tris / no original materials",fill=(0,0,0))
        im.save(OUT/f"block_lod{lod}.png")
        report[f"block_lod{lod}"]={"buildings":64,"triangles":len(m.faces),"material_groups":list(grouper.get_non_empty_groups()),"cpu_build_and_audit_ms":total_ms}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("CONDOR_REPORT "+json.dumps(report),flush=True)
    # A small genuine render, encoded for environments where artifact transfer is unavailable.
    thumb=sheet.copy();thumb.thumbnail((560,840))
    buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=60,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):
        print(f"CONDOR_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)
    assert len(report["cases"])==12
    for c in report["cases"]:
        if c["name"]=="courtyard":
            assert c["roof_coverage"]["hole_covered_area"]<1e-6, c
            assert c["roof_coverage"]["footprint_iou"]>0.99999, c
    # Report upstream defects explicitly; do not hide them in the preview.
    assert not any(c["degenerate_triangles"] for c in report["cases"]), "donor contains zero-area triangles"

if __name__=="__main__":main()
