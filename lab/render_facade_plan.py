"""Visual QA for actual aperture geometry, exported and reloaded from GLB."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pcg.footprint import Footprint
from pcg.massing import build_massing
from pcg.architecture import PROFILES, plan_facades
from pcg.facade_mesh import facade_triangles
from render_l2_massing import raster, VIEWS, SIZE

PALETTE={"wall":[205,192,168,255],"window":[45,78,104,255],
         "door":[90,58,36,255],"balcony_door":[45,78,104,255],"reveal":[143,132,117,255],
         "roof":[150,75,55,255],"base":[110,110,110,255]}


def render_mesh(f,plan,lod):
    h=plan.floor_levels[-1]
    shell=build_massing(f,h).shell
    roles={s.id:s.role for s in shell.surfaces}
    triangles=[(roles[k],tuple(shell.vertices[i] for i in t))
               for k,t in shell.triangles() if roles[k]!="wall"]
    for facade in plan.facades:
        triangles.extend((t.role,t.vertices) for t in facade_triangles(facade,h,lod))
    vertices=np.array([p for _,t in triangles for p in t])
    faces=np.arange(len(vertices)).reshape(-1,3)
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    mesh.visual.vertex_colors=np.repeat(np.array([PALETTE[r] for r,_ in triangles],dtype=np.uint8),3,axis=0)
    assert np.isfinite(vertices).all() and np.all(mesh.area_faces>0)
    return mesh


def main():
    out=Path("artifacts/l3-facade-plan");out.mkdir(parents=True,exist_ok=True)
    fixtures=[(p,6 if p=="house" else 9,2 if p=="house" else 3,False) for p in PROFILES]
    fixtures.append(("house",6,2,True))
    sheet=Image.new("RGB",(4*SIZE,len(fixtures)*2*SIZE),"white")
    report=[]
    for row,(profile,h,levels,yard) in enumerate(fixtures):
        name="courtyard" if yard else profile
        f=Footprint(name,[(0,0),(12,0),(12,10 if yard else 8),(0,10 if yard else 8)],
                    [[(3,3),(9,3),(9,7),(3,7)]] if yard else [])
        plan=plan_facades(f,h,levels,PROFILES[profile],party_edges=[(0,1)] if name=="apartments" else [])
        (out/(name+"_plan.json")).write_text(json.dumps(asdict(plan),indent=2))
        bounds=None
        for lod in (0,1):
            m=render_mesh(f,plan,lod)
            if bounds is None:bounds=m.bounds.copy()
            np.testing.assert_allclose(m.bounds,bounds,atol=1e-9)
            m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
            path=out/(f"{name}_lod{lod}.glb");path.write_bytes(m.export(file_type="glb"))
            loaded=trimesh.load(path,force="mesh",process=False)
            loaded.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            np.testing.assert_allclose(loaded.bounds,bounds,atol=2e-6)
            assert len(loaded.faces)==len(m.faces)
            for i,(label,view,up) in enumerate(VIEWS):
                canvas,_=raster(loaded,view,up)
                ImageDraw.Draw(canvas).text((10,10),f"{name} LOD{lod} / {label}",fill=(0,0,0))
                canvas.save(out/(f"{name}_lod{lod}_{label}.png"))
                sheet.paste(canvas,(i*SIZE,(row*2+lod)*SIZE))
            report.append({"name":name,"lod":lod,"plan_digest":plan.digest(),
                           "openings":sum(len(x.openings) for x in plan.facades),
                           "triangles":len(m.faces),"bounds":bounds.tolist()})
    sheet.save(out/"all_contact.png")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
