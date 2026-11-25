"""Actual GLB facade proxy diagnostics for the linear emission mip chain.

A single flat panel per selected facade, not a full-building far LOD.
The exact mask is from canonical glazing rectangles, before occlusion.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from pcg.architecture import PROFILES, plan_facades
from pcg.footprint import Footprint
from pcg.occupancy import plan_night
from pcg.far_emission import facade_emission, linear_mips, srgb_encode, srgb_decode
from render_l2_massing import raster

SIZE=384
RESOLUTIONS=(128,16,2,1)


def export_panel(facade,height,texture,path):
    verts=np.array([facade.point(u,z) for u,z in
                    ((0,0),(facade.length,0),(facade.length,height),(0,height))])
    uv=np.array([(0,0),(1,0),(1,1),(0,1)],dtype=float)
    mat=trimesh.visual.material.PBRMaterial(name="far_emission_proxy",
        baseColorTexture=Image.new("RGB",(2,2),(0,0,0)),
        emissiveTexture=texture,emissiveFactor=(1.,1.,1.),
        metallicFactor=0.,roughnessFactor=1.)
    mesh=trimesh.Trimesh(vertices=verts,faces=[(0,1,2),(0,2,3)],process=False,
        visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=mat))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
    path.write_bytes(mesh.export(file_type="glb"))
    scene=trimesh.load(path,force="scene",process=False)
    assert len(scene.geometry)==1
    loaded=next(iter(scene.geometry.values()))
    loaded.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    np.testing.assert_allclose(loaded.vertices,verts,atol=3e-6)
    np.testing.assert_allclose(loaded.visual.uv,uv,atol=1e-6)
    np.testing.assert_allclose(loaded.visual.material.emissiveFactor,(1,1,1))
    np.testing.assert_array_equal(np.asarray(loaded.visual.material.emissiveTexture.convert("RGB")),np.asarray(texture))
    np.testing.assert_allclose(loaded.area,facade.length*height,rtol=1e-6)
    return loaded


def main():
    out=Path("artifacts/l4-far-emission");out.mkdir(parents=True,exist_ok=True)
    fixture_path=Path("tests/fixtures/l4_neighborhood.json")
    fixture=json.loads(fixture_path.read_text())
    records=[];selected={}
    for c in sorted(fixture["buildings"],key=lambda c:c["id"]):
        f=Footprint(c["id"],c["outer"],c.get("holes",[]))
        h=c["eave"]
        plan=plan_facades(f,h,c["levels"],PROFILES[c["profile"]])
        intents=plan_night(plan,fixture["seed"])
        for facade in plan.facades:
            chain=linear_mips(facade_emission(facade,h,intents,128))
            expected=chain[0].mean((0,1))
            errors=[]
            for level in chain:
                q=np.round(srgb_encode(level)*255)/255
                error=float(np.max(np.abs(srgb_decode(q).mean((0,1))-expected)))
                assert error<=.0045
                errors.append(error)
            records.append({"facade_id":facade.id,"building":f.building_id,"profile":c["profile"],
                "area_m2":facade.length*h,"mean_linear":expected.tolist(),
                "max_float_mean_error":max(float(np.max(np.abs(level.mean((0,1))-expected))) for level in chain),
                "max_srgb8_mean_error":max(errors)})
            # Deterministically select the brightest facade of each profile,
            # to make the spatial changes visible in the diagnostic sheet.
            score=float(expected.sum())
            if c["profile"] not in selected or score>selected[c["profile"]][0]:
                selected[c["profile"]]=(score,facade,h,chain)
    sheet=Image.new("RGB",(len(RESOLUTIONS)*SIZE,len(selected)*SIZE))
    previews=[]
    for row,(profile,(_,facade,h,chain)) in enumerate(sorted(selected.items())):
        assert chain[0].max()>0
        initial_depth=None
        for col,n in enumerate(RESOLUTIONS):
            a=next(level for level in chain if level.shape[0]==n)
            tex=Image.fromarray(np.round(srgb_encode(a)*255).astype(np.uint8))
            key=profile+"_"+str(n)
            tex.save(out/(key+"_emissive.png"))
            path=out/(key+".glb")
            loaded=export_panel(facade,h,tex,path)
            center=facade.point(facade.length/2,h/2)
            im,depth=raster(loaded,(*facade.outward,0),(0,0,1),
                textured=True,night=True,center=center,pixels_per_m=20,size=SIZE)
            visible=np.isfinite(depth)
            assert visible.sum()>500
            canvas=np.asarray(im).copy()
            canvas[~visible]=[13,19,29]
            if initial_depth is None:
                initial_depth=depth.copy()
            else:
                np.testing.assert_array_equal(visible,np.isfinite(initial_depth))
                np.testing.assert_allclose(depth[visible],initial_depth[visible],atol=1e-6)
            # All views use the same geometry and physical camera scale.
            im=Image.fromarray(canvas)
            ImageDraw.Draw(im).text((8,8),profile+" / "+str(n)+" x "+str(n),fill=(230,230,235))
            im.save(out/(key+"_view.png"));sheet.paste(im,(col*SIZE,row*SIZE))
            previews.append({"profile":profile,"facade":facade.id,"size":n,
                             "glb_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    sheet.save(out/"all_contact.png")
    report={"schema_version":1,"compiler_version":"0.1.0",
            "fixture_sha256":hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "facades":records,"previews":previews,
            "max_float_mean_error":max(r["max_float_mean_error"] for r in records),
            "max_srgb8_mean_error":max(r["max_srgb8_mean_error"] for r in records),
            "limitations":["unoccluded aperture emission intent","flat diagnostic panels",
                           "separate mip GLBs, not runtime mip upload or LOD switching",
                           "no light transport, frames, rails, full HLOD or SDK"]}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"facades":len(records),"previews":len(previews),
        "max_float_mean_error":report["max_float_mean_error"],
        "max_srgb8_mean_error":report["max_srgb8_mean_error"]}))


if __name__=="__main__":
    main()
