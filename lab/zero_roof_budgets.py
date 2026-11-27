"""Research: actual GLB roof silhouette budgets on original synthetic archetypes.
Reuse canonical massing, opening plan, roof triangles and audited box primitive.
All roof accessories are opaque architectural volumes, never window geometry.
"""
from pathlib import Path
from collections import Counter
import base64
import copy
import io
import json
import math
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from pcg._grammar_boxes import _oriented_box
from pcg.footprint import Footprint
from pcg.massing import build_massing
from pcg.materials import MATERIALS,texture_maps
from pcg.far_emission import srgb_decode
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref

OUT=Path("artifacts/zero-roof")
BUDGETS=(0,36,96,192) # additional triangles per building, not targets
MATS={"chimney":"red_brick","vent":"metal_roof_red","access":"plaster_cream",
      "parapet":"concrete_panel","dormer":"ceramic_roof","cap":"bitumen_roof"}

def fixture():
    def rect(w,d):return [(0,0),(w,0),(w,d),(0,d)]
    rows=[("brick_low",24,12,2,"hip",2.8,"house"),
          ("brick_apartments",38,14,5,"gable",2.8,"apartments"),
          ("panel_tower",20,12,12,"flat",0,"apartments"),
          ("admin_block",28,16,6,"flat",0,"office")]
    return [dict(id=k,building_id="roof-fixture:"+k,outer=rect(w,d),holes=[],
                 floors=f,eave_m=3.*f,roof=roof,rise=rise,profile=p,
                 offset=[(i%2)*56.,(i//2)*40.,0.],party_edges=[])
            for i,(k,w,d,f,roof,rise,p) in enumerate(rows)]

def triangles(shell):
    v=np.asarray(shell.vertices,dtype=float)
    # Shell.triangles() yields (surface ID, indices), not (role, indices).
    roles={surface.id:surface.role for surface in shell.surfaces}
    out=np.asarray([v[list(t)] for sid,t in shell.triangles() if roles[sid]=="roof"])
    assert out.ndim==3 and out.shape[1:]==(3,3) and len(out)>0, shell.building_id
    return out

def roof_z(roof,x,y):
    # Exact barycentric support on the already generated piecewise planar roof.
    p=np.array([x,y]);zs=[]
    for t in roof:
        m=np.column_stack((t[1,:2]-t[0,:2],t[2,:2]-t[0,:2]))
        ab=np.linalg.solve(m,p-t[0,:2])
        if min(ab[0],ab[1],1-ab.sum())>=-1e-8:
            zs.append(float(t[0,2]+ab@(t[1:,2]-t[0,2])))
    assert zs and max(zs)-min(zs)<1e-7,(x,y,zs)
    return max(zs)

def box_mesh(cx,cy,w,d,h,z):
    v,faces=_oriented_box((cx,cy),(1.,0.),(0.,-1.),w,d,h,z)
    v=np.asarray(v,dtype=float)
    f=np.asarray([(q[0],q[j],q[j+1]) for q in faces for j in range(1,len(q)-1)])
    m=trimesh.Trimesh(vertices=v,faces=f,process=False)
    assert m.is_watertight and m.is_winding_consistent and m.volume>0
    return v[f]

def plan(c):
    f=Footprint(c["building_id"],c["outer"],c["holes"])
    shell=build_massing(f,c["eave_m"],c["roof"],c["rise"]).shell
    roof=triangles(shell);poly=f.polygon();x0,y0,x1,y1=poly.bounds
    # Independent roof projection coverage: a nonempty but incorrectly
    # filtered subset must fail too, including across a ridge.
    projected=[Polygon(t[:,:2]) for t in roof]
    coverage=unary_union(projected)
    assert coverage.symmetric_difference(poly).area<1e-7, c["id"]
    assert abs(sum(p.area for p in projected)-poly.area)<1e-7, c["id"]
    assert all(p.area>0 for p in projected)
    if c["roof"]=="flat":
        np.testing.assert_allclose(roof[:,:,2],c["eave_m"],rtol=0,atol=1e-9)
    else:
        np.testing.assert_allclose([roof[:,:,2].min(),roof[:,:,2].max()],
                                   [c["eave_m"],c["eave_m"]+c["rise"]],rtol=0,atol=1e-9)
    w=x1-x0;d=y1-y0
    modules=[];rejected=[];occupied=[]
    def add(kind,key,cx,cy,ww,dd,hh,cap=False):
        footprint=box(cx-ww/2,cy-dd/2,cx+ww/2,cy+dd/2)
        outer=footprint.buffer(.12,join_style=2) if cap else footprint
        if not poly.buffer(-.04).covers(outer):
            rejected.append((key,"outside_roof"));return
        if any(outer.intersection(p).area>1e-9 for p in occupied):
            rejected.append((key,"overlap"));return
        # Piecewise linear height extrema also occur at roof edges/vertices.
        # Intersect each roof triangle with the footprint and sample its vertices.
        z=[]
        for t in roof:
            hit=Polygon(t[:,:2]).intersection(outer)
            if hit.geom_type=="Polygon" and not hit.is_empty:
                z.extend(roof_z(roof,x,y) for x,y in list(hit.exterior.coords))
        assert z,(c["id"],key,"no roof support intersection")
        bottom=min(z)-.04;top=max(z)+hh
        geom=[(kind,box_mesh(cx,cy,ww,dd,top-bottom,bottom))]
        if cap:geom.append(("cap",box_mesh(cx,cy,ww+.24,dd+.24,.12,top)))
        module={"id":c["building_id"]+":roof:"+key,"kind":kind,"triangles":sum(len(t) for _,t in geom),
                "support_min_m":min(z),"support_max_m":max(z),"top_m":top,
                "footprint":list(outer.bounds),"geometry":geom}
        assert top>max(z) and bottom<min(z)
        modules.append(module);occupied.append(outer)
    # Ordered by hypothesized silhouette value, stable independent of budget.
    if c["roof"]=="flat":
        add("access","access",x0+.31*w,y0+.50*d,2.8,2.3,2.1)
        add("vent","vent0",x0+.70*w,y0+.62*d,1.1,1.1,.95,True)
        add("vent","vent1",x0+.70*w,y0+.32*d,.8,.8,.72,True)
        # Parapets only on these rectangular flat-roof fixtures. Four
        # non-overlapping closed sections; grouped as one budget decision.
        old=len(modules);t=.20
        for key,cx,cy,ww,dd in (
            ("south",x0+w/2,y0+.16,w-.12,t),("north",x0+w/2,y1-.16,w-.12,t),
            ("west",x0+.16,y0+d/2,t,d-.64),("east",x1-.16,y0+d/2,t,d-.64)):
            add("parapet","parapet_"+key,cx,cy,ww,dd,.48)
        parts=modules[old:]
        assert len(parts)==4
        modules[old:] = [dict(id=c["building_id"]+":roof:parapet",kind="parapet",
            triangles=sum(m["triangles"] for m in parts),
            geometry=[g for m in parts for g in m["geometry"]],sections=[{k:v for k,v in m.items() if k!="geometry"} for m in parts])]
    else:
        for j,fx in enumerate((.29,.71,.50)):
            add("chimney","chimney"+str(j),x0+fx*w,y0+.52*d,.72,.82,1.15,True)
        # Opaque dormer housing only, not a geometric window.
        # Reuse gabled massing for the next iteration; this fixture compares
        # simple service housings with support on the actual sloped roof.
        add("access","service_housing",x0+.48*w,y0+.22*d,2.6,1.7,1.25)
        add("vent","vent",x0+.20*w,y0+.25*d,.5,.5,.65,True)
    assert len({m["id"] for m in modules})==len(modules)
    return modules,rejected

def select(modules,budget):
    out=[];n=0
    for m in modules:
        if n+m["triangles"]>budget:break
        out.append(m);n+=m["triangles"]
    return out

def append(groups,table,c,m):
    # Reuse existing roof primitive and parameter role; no new material per asset.
    g=groups["roof"];offset=np.asarray(c["offset"],dtype=float)
    for j,(kind,tris) in enumerate(m["geometry"]):
        family=MATS[kind]
        for i,tri in enumerate(tris):
            sid=m["id"]+":"+str(j)+":"+str(i)
            rid=len(table);base=len(g["v"])
            n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
            tangent=np.cross([0.,0.,1.],n)
            if np.linalg.norm(tangent)<1e-10:tangent=np.array([1.,0.,0.])-n*n[0]
            tangent/=np.linalg.norm(tangent);bitangent=np.cross(n,tangent)
            uv=np.column_stack((tri@tangent,tri@bitangent))
            b=ref.pack([rid])[0]
            for k,val in (("v",tri+offset),("n",np.tile(n,(3,1))),("uv0",uv/MATERIALS[family].repeat_m),
                          ("uv1",uv),("color",np.tile([b[0],b[1],0,255],(3,1))),("f",[[base,base+1,base+2]])):
                g[k]=np.concatenate((g[k],np.asarray(val,dtype=g[k].dtype)))
            table.append(dict(id=rid,building=c["building_id"],surface=sid,role="roof",
                              family=family,seed=ref.seed(sid),height=c["eave_m"],floors=c["floors"]))

def main():
    OUT.mkdir(parents=True,exist_ok=True);cases=fixture()
    g0,t0,cells,owners=tile.compile_scene(cases,audit_samples=1024)
    # Regression: the original role/ID mixup returned an empty roof list.
    # Exercise each supported roof kind with an independent known center height.
    for c in cases:
        f=Footprint(c["building_id"],c["outer"],c["holes"])
        shell=build_massing(f,c["eave_m"],c["roof"],c["rise"]).shell
        r=triangles(shell);center=f.polygon().centroid
        assert abs(roof_z(r,center.x,center.y)-(c["eave_m"]+c["rise"]))<1e-7,c["id"]
    plans={c["id"]:plan(c) for c in cases}
    for c in cases:
        a,b=plans[c["id"]];aa,bb=plan(c)
        assert b==bb
        assert [m["id"] for m in a]==[m["id"] for m in aa]
        for m,n in zip(a,aa):
            for (_,x),(_,y) in zip(m["geometry"],n["geometry"]):np.testing.assert_array_equal(x,y)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in set(MATS.values())|{r["family"] for r in t0}}
    old=ref.shader
    def compose(sample,r,maps,mode,dense=False):
        col,em=old(sample,r,maps,"bare",False)
        if r["role"]!="wall":return col,em
        role,var,loc=tile.lookup(sample,r,cells)
        x,y=loc.T;op=(role==1)|(role==2)
        glass=op&(x>.10)&(x<.90)&(y>.10)&(y<.90)&(np.abs(x-.5)>.035)
        col[op]=srgb_decode(np.array([.78,.76,.70]))
        col[glass]=srgb_decode(np.column_stack((.15+.13*y[glass],.22+.14*y[glass],.27+.15*y[glass])))
        col[role==3]=srgb_decode(np.array([.32,.22,.14]))
        return col,em
    reports=[];previous=set();sheet=Image.new("RGB",(4*640,2*640+32),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual reloaded GLB / fixed cameras / CPU diffuse / base + roof accessory budgets",fill=(0,0,0))
    baseline_depth=None
    try:
        ref.shader=compose
        for col,budget in enumerate(BUDGETS):
            out=OUT/f"budget_{budget}";out.mkdir(exist_ok=True)
            groups=copy.deepcopy(g0);table=copy.deepcopy(t0);selected=[];counts={}
            for c in cases:
                mods=select(plans[c["id"]][0],budget)
                counts[c["id"]]=sum(m["triangles"] for m in mods)
                assert counts[c["id"]]<=budget
                for m in mods:
                    append(groups,table,c,m);selected.append({k:v for k,v in m.items() if k!="geometry"})
            ids={m["id"] for m in selected};assert previous<=ids;previous=ids
            assert table[:len(t0)]==t0
            for role in tile.GROUPS:
                for k,a in g0[role].items():np.testing.assert_array_equal(groups[role][k][:len(a)],a)
            mesh=out/"tile.glb";tile.export_glb(groups,mesh)
            v,uv,rids,roles=tile.read_glb(mesh,groups,table)
            shell_tri=sum(len(g["f"]) for g in g0.values())
            assert len(v)//3==shell_tri+sum(counts.values())
            (out/"parameters.json").write_text(json.dumps(table,indent=2))
            (out/"facade_cells.bin").write_bytes(cells.tobytes())
            (out/"placements.json").write_text(json.dumps(selected,indent=2))
            bounds=[v.min(0).tolist(),v.max(0).tolist()]
            # Fixed cameras across budgets, including elevated aerial view.
            depths=[]
            for row,(center,ppm,view) in enumerate((((46,27,15),4.6,(1,-1.5,.85)),((46,27,15),4.6,(.7,-1,2.8)))):
                im,dep=ref.render(v,uv,rids,table,maps,"day",center_override=center,ppm_override=ppm,view_override=view)
                im.save(out/f"view_{row}.png");depths.append(dep)
                ImageDraw.Draw(im).text((8,8),f"+{budget} max tris/building | {len(v)//3} total",fill=(0,0,0))
                sheet.paste(im,(col*640,row*640+32))
            if baseline_depth is None:baseline_depth=depths
            silhouette=[];changed=[]
            for a,b in zip(baseline_depth,depths):
                am=np.isfinite(a);bm=np.isfinite(b)
                assert np.all(~am|bm),"added geometry removes existing silhouette"
                silhouette.append(int(np.sum(am!=bm)))
                common=am&bm
                assert np.all(b[common]>=a[common]-1e-5),"added geometry behind prior visible depth"
                changed.append(int(np.sum(common&(b>a+1e-5))))
            reports.append(dict(budget=budget,triangles=len(v)//3,shell_triangles=shell_tri,
                added_triangles=sum(counts.values()),per_building_added=counts,modules=len(selected),
                module_kinds=dict(Counter(m["kind"] for m in selected)),glb_bytes=mesh.stat().st_size,
                bounds_local=bounds,silhouette_added_pixels=silhouette,depth_changed_pixels=changed,
                opening_triangles_added=0,canonical_table_prefix_exact=True))
        assert all(a["triangles"]<=b["triangles"] for a,b in zip(reports,reports[1:]))
        assert reports[-1]["added_triangles"]>0 and sum(reports[-1]["depth_changed_pixels"])>0
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report={"buildings":len(cases),"lanes":reports,"rejected":{k:v[1] for k,v in plans.items()},
            "scope":"Original synthetic rectangular archetypes; actual GLB reload; CPU diffuse diagnostic only",
            "limitations":["No pitched dormer roof, eave overhang, balcony or nonrectangular parapet yet",
                           "Opaque roof accessory boxes intentionally penetrate roof to avoid floating; not Boolean union or watertight building+accessory solid",
                           "No physical lighting/shadows, real photo textures, hardware timing or SDK",
                           "No snow, icicles or other weather/season effects; engine responsibility"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("ROOF_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1400,800));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=75,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"ROOF_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main()
