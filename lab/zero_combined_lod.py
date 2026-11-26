"""Combined roof/facade geometry LOD experiment. Research only.
Full reference retains EVERY balcony anchor. Compact face records preserve
PCGF v1 and reduce duplication, then simplify balcony envelope triangulation.
"""
from pathlib import Path
from collections import Counter
import base64,copy,hashlib,io,json,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import MultiPoint
import zero_facade_architecture as arch
import zero_roof_budgets as roof
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
from compile_zero_tiles import pack_table,LAYERS,opening_rectangles
from pcg.materials import MATERIALS,texture_maps
from pcg.far_emission import srgb_decode
from pcg.architecture import BALCONY_SLAB

OUT=Path("artifacts/zero-combined-lod")
LANES=("shell","full_reference","full_compact","all_anchor_lod")

def normal(t):
    n=np.cross(t[1]-t[0],t[2]-t[0]);return n/np.linalg.norm(n)

def roof_module(m):
    return dict(m,geometry=[(k,roof.MATS[k],t) for k,t in m["geometry"]])

def bounds(m):
    v=np.concatenate([t.reshape(-1,3) for _,_,t in m["geometry"]])
    return np.array([v.min(0),v.max(0)])

def simplify(m):
    if m["kind"] not in ("balcony_open","balcony_glazed"):
        return m
    slab=m["geometry"][0][2]
    lo,hi=slab.reshape(-1,3).min(0),slab.reshape(-1,3).max(0)
    # Merge the slab's vertical side strips with the envelope sides.
    # Keep slab top and bottom, all three exterior sides, and glazed cap.
    # Host-contact back face is intentionally omitted.
    horizontal=np.array([t for t in slab if abs(normal(t)[2])>.999])
    assert len(horizontal)==4
    out=[("slab",m["geometry"][0][1],horizontal)]
    for k,f,t in m["geometry"][1:]:
        q=t.copy()
        if k in ("rail","glazed"):
            z=q[:,:,2];bottom=z.min()
            assert abs(bottom-hi[2])<1e-8
            z[np.isclose(z,bottom,atol=1e-9,rtol=0)]=lo[2]
            # Distinct diagnostic flag restores material at original slab strip.
            k=k+"_merged"
        out.append((k,f,q))
    a=dict(m,geometry=out,triangles=sum(len(t) for _,_,t in out))
    assert a["triangles"]==m["triangles"]-8
    np.testing.assert_allclose(bounds(a),bounds(m),atol=1e-9,rtol=0)
    return a

def append_compact(groups,table,c,m):
    """One record per oriented face/part, not per triangle; indexed vertices
    shared within a record. Stable semantic ID derives from placement/part/face.
    """
    g=groups["roof"];off=np.asarray(c["offset"])
    for j,(kind,family,tris) in enumerate(m["geometry"]):
        batches={}
        for i,tri in enumerate(tris):
            n=normal(tri)
            key=tuple(np.round(np.r_[n,n@tri[0]],8))
            batches.setdefault(key,[]).append(i)
        for face,ii in enumerate(batches.values()):
            q=tris[ii];n=normal(q[0])
            assert all(np.allclose(normal(t),n,atol=1e-8,rtol=0) for t in q)
            tangent=np.cross([0.,0.,1.],n)
            if np.linalg.norm(tangent)<1e-10:tangent=np.array([1.,0.,0.])-n*n[0]
            tangent/=np.linalg.norm(tangent);bitangent=np.cross(n,tangent)
            allv=tris.reshape(-1,3);xy=np.column_stack((allv@tangent,allv@bitangent))
            span=np.maximum(xy.max(0)-xy.min(0),1e-6)
            unique,inv=np.unique(q.reshape(-1,3),axis=0,return_inverse=True)
            uv=np.column_stack((unique@tangent,unique@bitangent))-xy.min(0)
            sid=m["id"]+":"+str(j)+":face:"+str(face)
            rid=len(table);start=len(g["v"]);b=ref.pack([rid])[0]
            for k,val in (("v",unique+off),("n",np.tile(n,(len(unique),1))),
                          ("uv0",uv/MATERIALS[family].repeat_m),("uv1",uv),
                          ("color",np.tile([b[0],b[1],0,255],(len(unique),1))),
                          ("f",inv.reshape(-1,3)+start)):
                g[k]=np.concatenate((g[k],np.asarray(val,dtype=g[k].dtype)))
            table.append(dict(id=rid,building=c["building_id"],surface=sid,role="roof",family=family,
                seed=ref.seed(sid),height=c["eave_m"],floors=c["floors"],accessory=kind,
                span=span.tolist(),placement=m["id"]))

def shade_factory(old,cells,styles):
    base=arch.render_shader(old,cells,styles,True)
    def shade(sample,r,maps,mode,dense=False):
        k=r.get("accessory","")
        if k not in ("rail_merged","glazed_merged"):
            return base(sample,r,maps,mode,dense)
        # Recover original rail/glass metric coordinates, not rescale the
        # glazing to include the newly merged slab strip.
        q=dict(r,accessory=k.removesuffix("_merged"),
               span=[r["span"][0],r["span"][1]-BALCONY_SLAB])
        sample2=sample.copy();sample2[:,1]-=BALCONY_SLAB
        c,e=base(sample2,q,maps,mode,dense)
        strip=sample[:,1]<BALCONY_SLAB
        if strip.any():
            rr=dict(r,family="concrete_panel",accessory="slab")
            a,b=base(sample[strip],rr,maps,mode,dense);c[strip]=a;e[strip]=b
        return c,e
    return shade

def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True);cases=arch.fixtures()
    styles={c["building_id"]:c["id"] for c in cases}
    oldplan=tile.plan_facades
    try:
        tile.plan_facades=arch.styled_plan
        g0,t0,cells,owners=tile.compile_scene(cases,audit_samples=2048)
    finally:tile.plan_facades=oldplan
    plans={};anchor_ids=set()
    for c in cases:
        p,fac,fail=arch.plan_modules(c);rm,rf=roof.plan(c)
        assert p.digest()==next(b["plan_digest"] for b in owners if b["building_id"]==p.building_id)
        want={o.id for a in p.facades for o in a.openings if o.role=="balcony_door"}
        got={m["anchor"] for m in fac if m["kind"].startswith("balcony_")}
        assert got==want and len(got)==sum(m["kind"].startswith("balcony_") for m in fac)
        anchor_ids|=got
        # Roof modules are interior to host XY, facade modules outside; audit
        # actual projected convex hulls and vertical overlap independently.
        for a in fac:
            av=np.concatenate([t.reshape(-1,3) for _,_,t in a["geometry"]])
            ap=MultiPoint(av[:,:2]).convex_hull
            for b in rm:
                rb=roof_module(b);bv=np.concatenate([t.reshape(-1,3) for _,_,t in rb["geometry"]])
                bp=MultiPoint(bv[:,:2]).convex_hull
                assert ap.intersection(bp).area<1e-8 or min(av[:,2].max(),bv[:,2].max())<=max(av[:,2].min(),bv[:,2].min())+1e-8
        allmods=fac+[roof_module(m) for m in rm]
        assert len({m["id"] for m in allmods})==len(allmods)
        plans[c["id"]]=(allmods,fail,rf)
    assert len(anchor_ids)==sum(b["balcony_door_intents"] for b in owners)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    old=ref.shader;ref.shader=shade_factory(old,cells,styles)
    cameras=[((61,6,10),4.3,(.35,-1,.45)),((61,6,10),4.3,(.3,-1,1.8))]
    for c in cases:
        w=c["outer"][1][0];h=c["eave_m"]
        cameras.append(((c["offset"][0]+w/2,6,h/2),14.,(.30,-1,.40)))
    sheet=Image.new("RGB",(640*len(LANES),640*len(cameras)+32),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Combined actual GLB | fixed cameras | CPU diffuse | shell / full / compact / all-anchor LOD",fill=(0,0,0))
    reports=[];near=None;packed=None;near_arrays=None;near_table=None;near_ids=None
    try:
        for col,lane in enumerate(LANES):
            d=OUT/lane;d.mkdir(exist_ok=True);g=copy.deepcopy(g0);table=copy.deepcopy(t0);selected=[]
            for c in cases:
                for m in plans[c["id"]][0]:
                    if lane=="shell":continue
                    q=simplify(m) if lane=="all_anchor_lod" else m
                    if lane=="full_reference":arch.append(g,table,c,q)
                    else:append_compact(g,table,c,q)
                    selected.append({k:v for k,v in q.items() if k!="geometry"})
            if lane!="shell":
                assert {m["anchor"] for m in selected if m["kind"].startswith("balcony_")}==anchor_ids
            assert table[:len(t0)]==t0
            assert len({r["surface"] for r in table})==len(table)
            for role in tile.GROUPS:
                for k,a in g0[role].items():np.testing.assert_array_equal(g[role][k][:len(a)],a)
            mesh=d/"tile.glb";tile.export_glb(g,mesh);v,uv,ids,_=tile.read_glb(mesh,g,table)
            assert len(v)//3==sum(len(a["f"]) for a in g0.values())+sum(m["triangles"] for m in selected)
            if lane=="full_reference":
                near_arrays=(v.copy(),uv.copy());near_table=table;near_ids=ids.copy()
            if lane=="full_compact":
                # Face grouping preserves original triangle order for the
                # current shared box and quad recipes. Strict GLB equality.
                np.testing.assert_array_equal(v,near_arrays[0])
                np.testing.assert_allclose(uv,near_arrays[1],atol=1e-7,rtol=0)
                # Semantic/material equivalence for each original triangle,
                # irrespective of tile-local record index.
                for a,b in zip(near_ids[::3],ids[::3]):
                    x,y=near_table[int(a)],table[int(b)]
                    for k in ("building","role","family","accessory","span","placement"):
                        assert x.get(k)==y.get(k),(k,x,y)
            (d/"facade_cells.bin").write_bytes(cells.tobytes())
            samples=pack_table(table,cells,d/"facade_params.bin")
            check=json.loads(subprocess.run([decoder,str(d/"facade_params.bin"),str(d/"facade_cells.bin"),str(len(LAYERS))],
                capture_output=True,text=True,check=True).stdout)
            assert check["records"]==len(table) and check["cells"]==len(cells)
            (d/"parameters.json").write_text(json.dumps(table,indent=2))
            (d/"placements.json").write_text(json.dumps(selected,indent=2))
            images=[];depths=[]
            for row,(center,ppm,view) in enumerate(cameras):
                im,dep=ref.render(v,uv,ids,table,maps,"day",center_override=center,ppm_override=ppm,view_override=view)
                im.save(d/f"view_{row}.png");images.append(np.asarray(im).copy());depths.append(dep)
                ImageDraw.Draw(im).text((8,8),f"{lane} | {len(v)//3} tris / {len(table)} records",fill=(0,0,0))
                sheet.paste(im,(col*640,row*640+32))
            if lane=="full_reference":near=(images,depths)
            error=[]
            if near is not None:
                for a,b,da,db in zip(near[0],images,near[1],depths):
                    ma,mb=np.isfinite(da),np.isfinite(db);both=ma&mb
                    err=dict(silhouette_xor_pixels=int(np.sum(ma!=mb)),
                             max_common_depth_delta_m=float(np.max(np.abs(da[both]-db[both]))),
                             mean_rgb_abs_delta=float(np.abs(a.astype(float)-b).mean()))
                    if lane=="full_compact":
                        assert err["silhouette_xor_pixels"]==0
                        assert err["max_common_depth_delta_m"]<1e-7
                        assert err["mean_rgb_abs_delta"]<.01
                    error.append(err)
            # Canonical wall shading and emission unchanged for all budgets.
            for r in t0:
                if r["role"]!="wall":continue
                sample=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
                _,e0=arch.render_shader(old,cells,styles,False)(sample,r,maps,"night")
                _,e1=ref.shader(sample,table[r["id"]],maps,"night")
                np.testing.assert_array_equal(e0,e1)
            im,_=ref.render(v,uv,ids,table,maps,"night",center_override=cameras[0][0],ppm_override=cameras[0][1],view_override=cameras[0][2])
            im.save(d/"night.png")
            report=dict(lane=lane,triangles=len(v)//3,vertices=sum(len(a["v"]) for a in g.values()),
                records=len(table),parameter_bytes=(d/"facade_params.bin").stat().st_size,glb_bytes=mesh.stat().st_size,
                modules=len(selected),kinds=dict(Counter(m["kind"] for m in selected)),
                balcony_anchors=sum(m["kind"].startswith("balcony_") for m in selected),
                opening_triangles_added=0,abi_samples=samples,cpp_abi=True,
                cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest(),
                bounds=[v.min(0).tolist(),v.max(0).tolist()],vs_full_reference=error)
            reports.append(report)
        a,b,c,d=reports
        assert b["triangles"]==c["triangles"] and c["records"]<b["records"] and c["vertices"]<b["vertices"]
        assert c["parameter_bytes"]<b["parameter_bytes"] and c["glb_bytes"]<b["glb_bytes"]
        assert d["triangles"]==c["triangles"]-8*len(anchor_ids)
        assert d["balcony_anchors"]==c["balcony_anchors"]==len(anchor_ids)
        assert d["bounds"]==c["bounds"]
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report={"schema":"pcg.research.combined-lod.v1","buildings":owners,"lanes":reports,
        "planned_balcony_anchors":len(anchor_ids),"rejected":{k:{"facade":v[1],"roof":v[2]} for k,v in plans.items()},
        "limitations":["CPU diffuse synthetic fixtures, no native shader/GPU performance",
        "same PCGF v1 neutral role bucket; accessory style and facade zones still diagnostic sidecar",
        "open balcony envelope proxy, not collision, transparent glass or true recessed loggia",
        "all-anchor LOD removes slab vertical faces and merges side skins; measured image difference, no claim of exact visual match",
        "shell LOD omits visual balcony geometry; canonical intent/emission remains, not physical accessibility",
        "no runtime screen-space selection or far emission filtering in this experiment",
        "no weather/season effects, external texture assets or SDK"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("COMBINED_LOD_REPORT "+json.dumps(report),flush=True)
    thumb=Image.new("RGB",(640*3,640*3))
    for row in range(3):
        for col,lane in enumerate(("shell","full_compact","all_anchor_lod")):
            im=Image.open(OUT/lane/f"view_{row+2}.png")
            ImageDraw.Draw(im).text((8,8),lane,fill=(0,0,0));thumb.paste(im,(col*640,row*640))
    thumb.save(OUT/"close_contact.png");thumb.thumbnail((1200,1200))
    buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=73,optimize=True)
    s=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(s),4000):print(f"COMBINED_LOD_PREVIEW_{i//4000:03d} "+s[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
