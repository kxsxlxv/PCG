"""Export reusable asset resources and instantiated diagnostic GLBs."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pcg.assets import plan_assets, asset_triangles
from pcg.architecture import plan_facades, PROFILES
from pcg.footprint import Footprint
from render_facade_plan import render_mesh
from render_l2_massing import raster,VIEWS,SIZE

COLORS={"window_frame":[238,231,218,255],"door_frame":[155,120,87,255],
        "window_mullion":[238,231,218,255],"door_handle":[175,155,95,255],
        "balcony_slab":[183,173,156,255],"balcony_rail":[66,70,73,255],
        "balcony_bar":[66,70,73,255],"balcony_panel":[105,109,112,255]}


def mesh_for(tris):
    v=np.array([p for _,t in tris for p in t])
    m=trimesh.Trimesh(vertices=v,faces=np.arange(len(v)).reshape(-1,3),process=False)
    m.visual.vertex_colors=np.repeat(np.array([COLORS[r] for r,_ in tris],dtype=np.uint8),3,axis=0)
    return m


def export(mesh,path):
    mesh=mesh.copy()
    bounds=mesh.bounds.copy()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
    path.write_bytes(mesh.export(file_type="glb"))
    m=trimesh.load(path,force="mesh",process=False)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    np.testing.assert_allclose(m.bounds,bounds,atol=2e-6)
    assert len(m.faces)==len(mesh.faces)
    return m


def main():
    out=Path("artifacts/l3-assets");out.mkdir(parents=True,exist_ok=True)
    library=out/"library";library.mkdir(exist_ok=True)
    sheet=Image.new("RGB",(4*SIZE,6*SIZE),"white")
    report=[]
    emitted=set()
    for row,(profile,style) in enumerate(PROFILES.items()):
        f=Footprint(profile,[(0,0),(12,0),(12,8),(0,8)])
        plan=plan_facades(f,9 if profile!="house" else 6,3 if profile!="house" else 2,style,
                          party_edges=[(0,1)] if profile=="apartments" else [])
        assets=plan_assets(f,plan)
        recipes={r.id:r for r in assets.recipes}
        (out/(profile+"_placements.json")).write_text(json.dumps(asdict(assets),indent=2))
        bounds=None
        for lod in (0,1):
            cache={r.id:asset_triangles(r,lod) for r in assets.recipes}
            for key,tri in cache.items():
                filename=key.replace(":","_")+f"_lod{lod}.glb"
                if filename not in emitted:
                    export(mesh_for(tri),library/filename);emitted.add(filename)
            scene=render_mesh(f,plan,lod)
            placed=[(role,tuple(p.point(v) for v in tri)) for p in assets.placements for role,tri in cache[p.asset_id]]
            m=trimesh.util.concatenate([scene,mesh_for(placed)])
            if bounds is None:bounds=m.bounds.copy()
            np.testing.assert_allclose(m.bounds,bounds,atol=1e-9)
            loaded=export(m,out/(f"{profile}_lod{lod}.glb"))
            for i,(label,view,up) in enumerate(VIEWS):
                canvas,_=raster(loaded,view,up)
                ImageDraw.Draw(canvas).text((10,10),f"{profile} LOD{lod} / {label}",fill=(0,0,0))
                canvas.save(out/(f"{profile}_lod{lod}_{label}.png"))
                sheet.paste(canvas,(i*SIZE,(row*2+lod)*SIZE))
            report.append({"profile":profile,"lod":lod,"unique_assets":len(cache),
                           "placements":len(assets.placements),"rejected":list(assets.rejected),
                           "unique_asset_triangles":sum(map(len,cache.values())),
                           "expanded_asset_triangles":len(placed),"total_triangles":len(m.faces),
                           "bounds":bounds.tolist()})
    sheet.save(out/"all_contact.png")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
