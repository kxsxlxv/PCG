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
from pcg.occupancy import EMISSION,plan_night,mean_emission


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
    night_out=Path("artifacts/l4-night-occupancy");night_out.mkdir(parents=True,exist_ok=True)
    night_sheet=Image.new("RGB",(4*SIZE,3*SIZE),"white")
    night_report=[]
    for row,(profile,style) in enumerate(PROFILES.items()):
        f=Footprint(profile,[(0,0),(12,0),(12,8),(0,8)])
        h=6 if profile=="house" else 9
        plan=plan_facades(f,h,2 if profile=="house" else 3,style)
        assets=plan_assets(f,plan)
        shell=build_massing(f,h).shell
        roles={s.id:s.role for s in shell.surfaces}
        tris=[(roles[k],tuple(shell.vertices[i] for i in t),None) for k,t in shell.triangles() if roles[k]!="wall"]
        for s in plan.facades:tris.extend((t.role,t.vertices,t.opening_id) for t in facade_triangles(s,h,0))
        cache={r.id:asset_triangles(r,0) for r in assets.recipes}
        tris.extend((role,tuple(p.point(v) for v in t),None) for p in assets.placements for role,t in cache[p.asset_id])
        intents=plan_night(plan)
        variants={i.opening_id:i.variant for i in intents}
        groups={};night_groups={}
        for role,t,oid in tris:
            r=resolve_material(profile,role)
            groups.setdefault(r.id,[]).append(t)
            variant=variants[oid] if role in ("window","balcony_door") else "off"
            key=r.id if variant=="off" else r.id+"__"+variant
            night_groups.setdefault(key,[]).append(t)
        # Same original triangles, grouped by a bounded emissive variant only.
        assert sum(len(g) for g in night_groups.values())==len(tris)
        assert set(night_groups)<=set(groups)|{"glass_proxy__warm","glass_proxy__neutral"}
        day_depth={}
        for night,active_groups,output,contact in ((False,groups,out,sheet),(True,night_groups,night_out,night_sheet)):
            scene=trimesh.Scene()
            for key,tri in sorted(active_groups.items()):
                base,_,variant=key.partition('__')
                r=MATERIALS[base];m=maps[base]
                v=np.array(tri).reshape(-1,3)
                uv=np.concatenate([metric_uv(t,r.repeat_m)[0] for t in tri])
                normals=np.repeat(np.array([metric_uv(t,r.repeat_m)[1] for t in tri]),3,axis=0)
                mat=trimesh.visual.material.PBRMaterial(name=key,baseColorTexture=m["basecolor"],
                    normalTexture=m["normal"],metallicRoughnessTexture=m["orm"],occlusionTexture=m["orm"],
                    metallicFactor=1.,roughnessFactor=1.,
                    emissiveFactor=EMISSION[variant or 'off'] if night else None)
                mesh=trimesh.Trimesh(vertices=v,faces=np.arange(len(v)).reshape(-1,3),
                     vertex_normals=normals,process=False,
                     visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=mat))
                mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
                scene.add_geometry(mesh,geom_name=key,node_name=key)
            path=output/(profile+".glb");path.write_bytes(scene.export(file_type="glb"))
            loaded=trimesh.load(path,force="scene",process=False)
            meshes=list(loaded.geometry.values())
            assert len(meshes)==len(active_groups)
            assert sum(len(m.faces) for m in meshes)==len(tris)
            for m in meshes:
                key=m.visual.material.name
                assert key in active_groups
                base,_,variant=key.partition('__')
                if night:
                    np.testing.assert_allclose(m.visual.material.emissiveFactor if m.visual.material.emissiveFactor is not None else (0,0,0),EMISSION[variant or 'off'],atol=1e-6)
                assert np.isfinite(m.visual.uv).all()
                np.testing.assert_array_equal(np.asarray(m.visual.material.baseColorTexture.convert("RGB")),np.asarray(maps[base]["basecolor"]))
                assert m.visual.material.normalTexture is not None
                assert m.visual.material.metallicRoughnessTexture is not None
                m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
                np.testing.assert_allclose(m.vertices,np.array(active_groups[key]).reshape(-1,3),atol=2e-6)
                expected=np.concatenate([metric_uv(t,MATERIALS[base].repeat_m)[0] for t in active_groups[key]])
                np.testing.assert_allclose(m.visual.uv,expected,atol=2e-6)
            for col,(label,view,up) in enumerate(VIEWS):
                depth=np.full((SIZE,SIZE),-np.inf)
                canvas=np.full((SIZE,SIZE,3),[13,19,29] if night else [245,243,238],dtype=np.uint8)
                for m in meshes:
                    im,z=raster(m,view,up,textured=True,night=night)
                    mask=z>depth
                    canvas[mask]=np.asarray(im)[mask]
                    depth[mask]=z[mask]
                assert np.isfinite(depth).sum()>100
                if night:
                    np.testing.assert_array_equal(np.isfinite(depth),np.isfinite(day_depth[label]))
                    np.testing.assert_allclose(depth[np.isfinite(depth)],day_depth[label][np.isfinite(depth)],atol=2e-6)
                else:
                    day_depth[label]=depth.copy()
                im=Image.fromarray(canvas)
                ImageDraw.Draw(im).text((8,8),profile+" / "+label,fill=(220,225,235) if night else (0,0,0))
                im.save(output/(profile+"_"+label+".png"));contact.paste(im,(col*SIZE,row*SIZE))
            if not night:
                day_bytes=path.stat().st_size
            else:
                night_report.append({"profile":profile,"windows":len(intents),
                    "lit":sum(i.variant!="off" for i in intents),"mean_linear_glazing":mean_emission(intents),
                    "materials":len(active_groups),"triangles":len(tris),"glb_bytes":path.stat().st_size,
                    "plan_digest":plan.digest(),"intents":[asdict(i) for i in intents]})
        report.append({"profile":profile,"materials":len(groups),"triangles":len(tris),"glb_bytes":day_bytes})
    night_sheet.save(night_out/"all_contact.png")
    (night_out/"report.json").write_text(json.dumps(night_report,indent=2))
    print(json.dumps([{k:v for k,v in item.items() if k!="intents"} for item in night_report]))
    sheet.save(out/"all_contact.png")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
