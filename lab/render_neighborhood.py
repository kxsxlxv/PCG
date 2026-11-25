"""L4 synthetic visual block: existing PCG generators, shared PBR groups.

Orthographic diagnostics, not a GIS source, real street camera or PBR renderer.
The fixed fixture uses flat/convex-hip roofs; gable facade integration is deferred.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw
from shapely import contains_xy, distance, points
from shapely.geometry import box
from shapely.ops import unary_union

from pcg.footprint import Footprint
from pcg.architecture import PROFILES, plan_facades
from pcg.facade_mesh import facade_triangles
from pcg.massing import build_massing
from pcg.assets import plan_assets, asset_triangles
from pcg.materials import MATERIALS, resolve_material
from pcg.occupancy import plan_night, mean_emission
from render_material_uv import export_reload
from render_l2_massing import raster

SIZE = 768
SCALE = 10.5
CENTER = (26.,17.,6.)
VIEWS = (("top",(0,0,1),(0,1,0)),
         ("aerial",(1,-1,1.4),(0,0,1)),
         ("low_southwest",(-.35,-1,.16),(0,0,1)),
         ("low_northeast",(1,.65,.22),(0,0,1)))


def build_block(fixture):
    assert fixture["schema_version"]==1 and fixture["frame"]=="local_z_up" and fixture["units"]=="m"
    cases=sorted(fixture["buildings"],key=lambda c:c["id"])
    if len({c["id"] for c in cases})!=len(cases):
        raise ValueError("duplicate building id")
    day={};night={};records=[];envelopes=[];polygons=[]
    for c in cases:
        profile=c["profile"]
        f=Footprint(c["id"],c["outer"],c.get("holes",[]))
        h=c["eave"]
        if c["roof"] not in ("flat","hip"):
            raise ValueError("block supports flat and convex hip facade integration")
        massing=build_massing(f,h,c["roof"],c["rise"])
        # This specific fixture promises these roof types, not a silent fallback.
        assert massing.actual_roof==c["roof"], (f.building_id,massing.fallback_reason)
        shell=massing.shell
        plan=plan_facades(f,h,c["levels"],PROFILES[profile])
        assets=plan_assets(f,plan)
        roles={s.id:s.role for s in shell.surfaces}
        tris=[(roles[k],tuple(shell.vertices[i] for i in t),None)
              for k,t in shell.triangles() if roles[k]!="wall"]
        for facade in plan.facades:
            tris.extend((t.role,t.vertices,t.opening_id)
                        for t in facade_triangles(facade,h,0))
        cache={r.id:asset_triangles(r,0) for r in assets.recipes}
        tris.extend((role,tuple(p.point(v) for v in t),None)
                    for p in assets.placements for role,t in cache[p.asset_id])
        vertices=np.array([t for _,t,_ in tris]).reshape(-1,3)
        assert np.isfinite(vertices).all()
        lo=vertices.min(0);hi=vertices.max(0)
        np.testing.assert_allclose([lo[2],hi[2]],[0,h+c["rise"]],atol=1e-8)
        envelope=box(lo[0],lo[1],hi[0],hi[1])
        # Conservative separation of actual render bounds including protrusions.
        assert all(not envelope.intersects(other) for other in envelopes),f.building_id
        envelopes.append(envelope);polygons.append(f.polygon())
        intents=plan_night(plan,fixture["seed"])
        variants={i.opening_id:i.variant for i in intents}
        for role,t,oid in tris:
            key=resolve_material(profile,role).id
            if role=="roof" and massing.actual_roof=="hip":
                key="ceramic_roof"
            day.setdefault(key,[]).append(t)
            variant=variants[oid] if role in ("window","balcony_door") else "off"
            nk=key if variant=="off" else key+"__"+variant
            night.setdefault(nk,[]).append(t)
        records.append({"id":f.building_id,"profile":profile,"footprint_digest":f.digest(),
                        "plan_digest":plan.digest(),"area_m2":f.polygon().area,
                        "roof":massing.actual_roof,"fallback":massing.fallback_reason,
                        "bounds_z_up":[lo.tolist(),hi.tolist()],"triangles":len(tris),
                        "windows":len(intents),"lit":sum(i.variant!="off" for i in intents),
                        "mean_linear_glazing":mean_emission(intents),"asset_placements":len(assets.placements)})
    assert sum(len(g) for g in day.values())==sum(len(g) for g in night.values())==sum(r["triangles"] for r in records)
    assert set(night)<=set(day)|{"glass_proxy__warm","glass_proxy__neutral"}
    return day,night,records,unary_union(polygons)


def main():
    out=Path("artifacts/l4-neighborhood");out.mkdir(parents=True,exist_ok=True)
    fixture_path=Path("tests/fixtures/l4_neighborhood.json")
    fixture=json.loads(fixture_path.read_text())
    day,night,records,poly=build_block(fixture)
    # Stable grouping under input order reversal, including window identity.
    again=build_block({**fixture,"buildings":list(reversed(fixture["buildings"]))})
    assert (day,night,records)==again[:3]
    maps={}
    for base in sorted(day):
        maps[base]={}
        for channel in ("basecolor","normal","orm"):
            with Image.open(Path("artifacts/l4-material-uv")/base/(channel+".png")) as im:
                maps[base][channel]=im.copy()
    sheet=Image.new("RGB",(2*SIZE,4*SIZE))
    day_depth={};results={}
    for column,(name,groups,is_night) in enumerate((("day",day,False),("night",night,True))):
        path=out/(name+".glb")
        meshes=export_reload(groups,maps,path,night=is_night)
        all_vertices=np.vstack([m.vertices for m in meshes])
        results[name]={"materials":len(groups),"triangles":sum(len(m.faces) for m in meshes),
                       "glb_bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                       "views":{}}
        for row,(label,view,up) in enumerate(VIEWS):
            direction=np.asarray(view,dtype=float);direction/=np.linalg.norm(direction)
            right=np.cross(up,direction);right/=np.linalg.norm(right)
            vertical=np.cross(direction,right)
            vv=all_vertices-np.asarray(CENTER)
            xy=np.column_stack((vv@right,-vv@vertical))*SCALE+SIZE/2
            assert xy.min()>12 and xy.max()<SIZE-12,(label,xy.min(),xy.max())
            depth=np.full((SIZE,SIZE),-np.inf)
            canvas=np.full((SIZE,SIZE,3),[13,19,29] if is_night else [245,243,238],dtype=np.uint8)
            for m in meshes:
                im,z=raster(m,view,up,textured=True,night=is_night,
                            center=CENTER,pixels_per_m=SCALE,size=SIZE)
                mask=z>depth
                canvas[mask]=np.asarray(im)[mask]
                depth[mask]=z[mask]
            visible=np.isfinite(depth)
            assert visible.sum()>500
            if label=="top":
                yy,xx=np.mgrid[0:SIZE,0:SIZE]
                x=(xx+.5-SIZE/2)/SCALE+CENTER[0]
                y=-(yy+.5-SIZE/2)/SCALE+CENTER[1]
                expected=contains_xy(poly,x,y)
                # Ignore <=0.85m from the outline: real balcony protrusions,
                # facade trim and pixel-boundary discretization live there.
                safe=distance(poly.boundary,points(x,y))>.85
                assert not np.any(safe & (visible!=expected)),name
            if is_night:
                np.testing.assert_array_equal(visible,np.isfinite(day_depth[label]))
                np.testing.assert_allclose(depth[visible],day_depth[label][visible],atol=1e-5)
            else:
                day_depth[label]=depth.copy()
            im=Image.fromarray(canvas)
            ImageDraw.Draw(im).text((12,12),name+" / "+label,fill=(230,230,235) if is_night else (0,0,0))
            im.save(out/(name+"_"+label+".png"))
            sheet.paste(im,(column*SIZE,row*SIZE))
            results[name]["views"][label]={"visible_pixels":int(visible.sum())}
    sheet.save(out/"all_contact.png")
    report={"schema_version":1,"compiler_version":"0.1.0","fixture_sha256":hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "camera":{"center":CENTER,"pixels_per_m":SCALE,"size":SIZE,"views":VIEWS},
            "buildings":records,"day_night":results,"limitations":["synthetic layout","orthographic CPU diagnostics",
            "no ground/roads, actual terrain, PBR lighting or native performance","no far-emission mip or HLOD"]}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"neighborhood_buildings":len(records),"day":results["day"],"night":results["night"]}))


if __name__=="__main__":
    main()
