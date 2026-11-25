"""Metric UV and shared PBR GLB export baseline, with actual reloaded texture preview."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pcg.footprint import Footprint
from pcg.architecture import PROFILES, plan_facades
from pcg.facade_mesh import facade_triangles
from pcg.massing import build_massing
from pcg.assets import plan_assets,asset_triangles
from pcg.materials import MATERIALS,resolve_material,metric_uv,texture_maps
from render_l2_massing import raster,VIEWS,SIZE


def image(a):
    return Image.fromarray(np.round(a*255).astype(np.uint8))


def main():
    out=Path("artifacts/l4-material-uv");out.mkdir(parents=True,exist_ok=True)
    maps={};manifest={}
    for key,r in MATERIALS.items():
        d=out/key;d.mkdir(exist_ok=True)
        maps[key]={}
        records={}
        for name,a in texture_maps(r).items():
            im=image(a);path=d/(name+".png");im.save(path)
            np.testing.assert_array_equal(np.asarray(Image.open(path)),np.asarray(im))
            maps[key][name]=im
            records[name]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
        manifest[key]={"recipe":asdict(r),"digest":r.digest(),"maps":records,
            "resolution":256,"seed":17,"colorspace":{"basecolor":"sRGB","normal":"linear","orm":"linear","height":"linear"},
            "normal_convention":"OpenGL +Y","orm_channels":["occlusion","roughness","metalness"]}
    (out/"materials.json").write_text(json.dumps(manifest,indent=2))
    sheet=Image.new("RGB",(4*SIZE,3*SIZE),"white")
    report=[]
    for row,(profile,style) in enumerate(PROFILES.items()):
        f=Footprint(profile,[(0,0),(12,0),(12,8),(0,8)])
        h=6 if profile=="house" else 9
        plan=plan_facades(f,h,2 if profile=="house" else 3,style)
        assets=plan_assets(f,plan)
        shell=build_massing(f,h).shell
        roles={s.id:s.role for s in shell.surfaces}
        tris=[(roles[k],tuple(shell.vertices[i] for i in t)) for k,t in shell.triangles() if roles[k]!="wall"]
        for s in plan.facades:tris.extend((t.role,t.vertices) for t in facade_triangles(s,h,0))
        cache={r.id:asset_triangles(r,0) for r in assets.recipes}
        tris.extend((role,tuple(p.point(v) for v in t)) for p in assets.placements for role,t in cache[p.asset_id])
        groups={}
        for role,t in tris:
            r=resolve_material(profile,role)
            groups.setdefault(r.id,[]).append(t)
        scene=trimesh.Scene()
        for key,tri in sorted(groups.items()):
            r=MATERIALS[key];m=maps[key]
            v=np.array(tri).reshape(-1,3)
            uv=np.concatenate([metric_uv(t,r.repeat_m)[0] for t in tri])
            normals=np.repeat(np.array([metric_uv(t,r.repeat_m)[1] for t in tri]),3,axis=0)
            mat=trimesh.visual.material.PBRMaterial(name=key,baseColorTexture=m["basecolor"],
                normalTexture=m["normal"],metallicRoughnessTexture=m["orm"],occlusionTexture=m["orm"],
                metallicFactor=1.,roughnessFactor=1.)
            mesh=trimesh.Trimesh(vertices=v,faces=np.arange(len(v)).reshape(-1,3),
                 vertex_normals=normals,process=False,
                 visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=mat))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
            scene.add_geometry(mesh,geom_name=key,node_name=key)
        path=out/(profile+".glb");path.write_bytes(scene.export(file_type="glb"))
        loaded=trimesh.load(path,force="scene",process=False)
        meshes=list(loaded.geometry.values())
        assert len(meshes)==len(groups)
        assert sum(len(m.faces) for m in meshes)==len(tris)
        for m in meshes:
            key=m.visual.material.name
            assert key in groups
            assert np.isfinite(m.visual.uv).all()
            np.testing.assert_array_equal(np.asarray(m.visual.material.baseColorTexture.convert("RGB")),np.asarray(maps[key]["basecolor"]))
            assert m.visual.material.normalTexture is not None
            assert m.visual.material.metallicRoughnessTexture is not None
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
            expected=np.concatenate([metric_uv(t,MATERIALS[key].repeat_m)[0] for t in groups[key]])
            np.testing.assert_allclose(m.visual.uv,expected,atol=2e-6)
        for col,(label,view,up) in enumerate(VIEWS):
            depth=np.full((SIZE,SIZE),-np.inf)
            canvas=np.full((SIZE,SIZE,3),[245,243,238],dtype=np.uint8)
            for m in meshes:
                im,z=raster(m,view,up,textured=True)
                mask=z>depth
                canvas[mask]=np.asarray(im)[mask]
                depth[mask]=z[mask]
            assert np.isfinite(depth).sum()>100
            im=Image.fromarray(canvas)
            ImageDraw.Draw(im).text((8,8),profile+" / "+label,fill=(0,0,0))
            im.save(out/(profile+"_"+label+".png"));sheet.paste(im,(col*SIZE,row*SIZE))
        report.append({"profile":profile,"materials":len(groups),"triangles":len(tris),"glb_bytes":path.stat().st_size})
    sheet.save(out/"all_contact.png")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
