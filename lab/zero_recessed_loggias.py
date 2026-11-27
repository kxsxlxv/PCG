"""Research 16: actual recessed architectural stacks, canonical PCGF v1.
One continuous recess per selected facade, not one cutout per opening.
Reuse existing canonical compiler, Shapely CDT, facade frames, balcony
envelope shading, compact accessory exporter and C++ numeric audit.
"""
from pathlib import Path
from collections import Counter
import base64,copy,hashlib,io,json,sys,subprocess
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from pcg.materials import MATERIALS,texture_maps
from pcg.architecture import BALCONY_SLAB
from compile_zero_tiles import pack_table,LAYERS,opening_rectangles
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_facade_architecture as arch
import zero_roof_budgets as roof
import zero_combined_lod as combined
import zero_architecture_correctness as corr
import zero_roof_construction as construction
import zero_balcony_guard_variants as variants

OUT=Path("artifacts/zero-recessed-loggias")
CLEAR=.05
DEPTHS=(0.,.6,1.)

def oriented_tri(v,uv,n):
    v=np.asarray(v,float);uv=np.asarray(uv,float)
    cross=np.cross(v[1]-v[0],v[2]-v[0])
    assert np.linalg.norm(cross)>1e-8
    if cross@n<0:v=v[[0,2,1]];uv=uv[[0,2,1]]
    assert np.dot(combined.normal(v),n)>.999999
    return v,uv

def quad(v,n):
    v=np.asarray(v,float)
    return np.array([oriented_tri(v[i],np.zeros((3,2)),n)[0] for i in ([0,1,2],[0,2,3])])

def frame(a):
    t=(np.asarray(a.end)-a.start)/a.length
    n=np.asarray(a.outward);up=np.array([0.,0.,1.])
    tangent=np.r_[t,0.];normal=np.r_[n,0.];origin=np.r_[a.start,0.]
    np.testing.assert_allclose(np.cross(tangent,up),normal,atol=1e-9)
    def pos(uv,depth=0.):
        uv=np.asarray(uv,float)
        return origin+uv[:,0,None]*tangent+uv[:,1,None]*up-depth*normal
    return pos,tangent,normal,up

def plan_recesses(c,p,mods,table,cells):
    if c["id"]!="panel_block":return []
    result=[]
    for a in p.facades:
        r=next(r for r in table if r.get("facade_id")==a.id)
        rects=opening_rectangles(r,cells)
        if a.length<20:continue
        pos,t,n,up=frame(a)
        # One complete six-storey balcony stack on each long face.
        slots=[(j,b) for j,(code,b) in rects.items()
               if j<r["floors"]*r["bays"] and j%r["bays"]==0 and code&3==2]
        assert len(slots)==r["floors"]-1==6
        mids=[m for m in mods if m.get("facade")==a.id and m["kind"].startswith("balcony_") and m["anchor"].endswith(":0")]
        assert len(mids)==len(slots)
        mids.sort(key=lambda m:combined.bounds(m)[0,2])
        selected={m["id"] for m in mids}
        assert len(selected)==6
        for (j,b),m in zip(slots,mids):
            assert m["anchor"].endswith(":"+str(j//r["bays"])+":0")
            assert abs(combined.bounds(m)[0,2]+BALCONY_SLAB-b[1])<1e-8
        x=min(b[0] for _,b in slots)-.30;X=max(b[2] for _,b in slots)+.30
        z=min(combined.bounds(m)[0,2] for m in mids)
        Z=max(max(b[3] for _,b in slots)+.18,max(combined.bounds(m)[1,2] for m in mids)+.10)
        cut=box(x,z,X,Z);wall=box(0,0,r["length"],r["height"])
        assert wall.buffer(-CLEAR).covers(cut)
        for j,(_,b) in rects.items():
            aperture=box(*b)
            if j in {k for k,_ in slots}:assert cut.buffer(-CLEAR+1e-8).covers(aperture)
            else:assert cut.intersection(aperture.buffer(CLEAR-1e-8,join_style=2)).area<1e-9
        result.append(dict(r=r,a=a,cut=cut,selected=selected,modules=mids,slots=slots))
    assert len(result)==2
    assert len(set.union(*(q["selected"] for q in result)))==12
    return result

def rebuild_wall(groups,r,uvtris,pos,depths,normal,offset):
    g=groups["wall"];idx=ref.decode(g["color"][:,:2].astype(np.float32)/255)
    faces=g["f"];mask=idx[faces[:,0]]==r["id"]
    assert mask.sum()==2 and np.all(idx[faces[mask]]==r["id"])
    keep=faces[~mask];used,inv=np.unique(keep.reshape(-1),return_inverse=True)
    for k in g:
        if k!="f":g[k]=g[k][used]
    g["f"]=inv.reshape(-1,3).astype(g["f"].dtype)
    # Share vertices with identical position and layout within this record.
    v=np.concatenate([pos(u,d) for u,d in zip(uvtris,depths)])
    u=np.concatenate(uvtris)
    unique,inverse=np.unique(np.column_stack((v,u)),axis=0,return_inverse=True)
    v=unique[:,:3];u=unique[:,3:];start=len(g["v"]);b=ref.pack([r["id"]])[0]
    for k,val in (("v",v+offset),("n",np.tile(normal,(len(v),1))),
                  ("uv0",u/MATERIALS[r["family"]].repeat_m),("uv1",u),
                  ("color",np.tile([b[0],b[1],0,255],(len(v),1))),
                  ("f",inverse.reshape(-1,3)+start)):
        g[k]=np.concatenate((g[k],np.asarray(val,dtype=g[k].dtype)))

def recess(groups,table,c,q,depth):
    r,a,cut=q["r"],q["a"],q["cut"];x,z,X,Z=cut.bounds
    pos,t,n,up=frame(a)
    outer=box(0,0,r["length"],r["height"]).difference(cut)
    pieces=list(constrained_delaunay_triangles(outer).geoms)
    assert unary_union(pieces).symmetric_difference(outer).area<1e-9
    assert abs(sum(p.area for p in pieces)-outer.area)<1e-9
    uvtris=[];depths=[]
    for p in pieces:
        uv=np.asarray(p.exterior.coords[:-1]);_,u=oriented_tri(pos(uv),uv,n)
        uvtris.append(u);depths.append(0.)
    uv=np.array([[x,z],[X,z],[X,Z],[x,Z]])
    for i in ([0,1,2],[0,2,3]):
        _,u=oriented_tri(pos(uv[i],depth),uv[i],n);uvtris.append(u);depths.append(depth)
    rebuild_wall(groups,r,uvtris,pos,depths,n,np.asarray(c["offset"]))
    front=pos(uv);back=pos(uv,depth)
    returns=[]
    for i,want in enumerate((up,-t,-up,t)):
        j=(i+1)%4
        returns.append(("loggia_reveal",r["family"],quad([front[i],front[j],back[j],back[i]],want)))
    reveal=dict(id=a.id+":recess",kind="loggia_reveal",geometry=returns,
        triangles=8,depth_m=depth,cut_uv_m=list(cut.bounds),removed_volume_m3=cut.area*depth)
    # Architectural slab skins span the cavity. Their sides meet the
    # host reveals; omit buried side/back faces, and the first bottom
    # face on the existing cavity floor, to avoid duplicate coplanar faces.
    mods=[]
    def horizontal(z0,want):
        uv=np.array([[x,z0],[X,z0]])
        f=pos(uv);b=pos(uv,depth)
        return quad([f[0],f[1],b[1],b[0]],want)
    def facade_strip(z0,z1,u0=x,u1=X):
        return quad(pos(np.array([[u0,z0],[u1,z0],[u1,z1],[u0,z1]])),n)
    for m in q["modules"]:
        lo,hi=combined.bounds(m);base=lo[2];sill=base+BALCONY_SLAB
        assert z-1e-8<=base<sill<hi[2]<=Z+1e-8
        parts=[("slab","concrete_panel",horizontal(sill,up)),
               ("slab","concrete_panel",facade_strip(base,sill))]
        if base>z+1e-8:parts.append(("slab","concrete_panel",horizontal(base,-up)))
        family=m["envelope_variant"]
        if family=="open":
            parts.append(("rail","painted_rail",facade_strip(sill,hi[2])))
        elif family=="full":
            parts.append(("glazed","painted_rail",facade_strip(sill,hi[2])))
        else:
            assert family=="partial"
            middle=sill+.38*(hi[2]-sill)
            parts.extend([("partial_lower","painted_rail",facade_strip(sill,middle)),
                          ("partial_upper","painted_rail",facade_strip(middle,hi[2],x,(x+X)/2))])
        new=dict(m,geometry=parts,kind="recessed_loggia",depth_m=depth,
                 triangles=sum(len(t0) for _,_,t0 in parts))
        new["bounds"]=combined.bounds(new).tolist()
        verts=np.concatenate([t0.reshape(-1,3) for _,_,t0 in parts])
        dist=(verts-np.r_[a.start,0.])@n
        assert dist.max()<1e-8 and dist.min()>=-depth-1e-8
        mods.append(new)
    return reveal,mods

def host_mesh(v,ids,table,bid):
    keep=np.array([table[int(i)]["building"]==bid and
         ("accessory" not in table[int(i)] or table[int(i)]["accessory"]=="loggia_reveal")
         for i in ids[::3]])
    tris=v.reshape(-1,3,3)[keep]
    m=trimesh.Trimesh(vertices=tris.reshape(-1,3),faces=np.arange(tris.size//3).reshape(-1,3),process=True)
    assert m.is_watertight and m.is_winding_consistent and m.volume>0
    construction.require_connected_faces(m)
    return m,len(tris)

def audit_projection(v,uv,ids,q,cells,depth,offset):
    r,a,cut=q["r"],q["a"],q["cut"];pos,t,n,up=frame(a)
    mask=ids[::3]==r["id"];verts=v.reshape(-1,3,3)[mask];layout=uv.reshape(-1,3,2)[mask]
    dist=(verts-np.r_[a.start,0.]-offset)@n
    planes=[];polys=[]
    for xyz,tri in zip(dist,layout):
        assert np.ptp(xyz)<1e-6
        assert min(abs(xyz[0]),abs(xyz[0]+depth))<1e-6
        polys.append(Polygon(tri));planes.append(float(xyz[0]))
    wall=box(0,0,r["length"],r["height"])
    assert unary_union(polys).symmetric_difference(wall).area<1e-5
    assert abs(sum(p.area for p in polys)-wall.area)<1e-5
    if depth:
        recessed=unary_union([p for p,d in zip(polys,planes) if d<-depth/2])
        assert recessed.symmetric_difference(cut).area<1e-5
    for j,(_,b) in opening_rectangles(r,cells).items():
        aperture=box(*b)
        expect=-depth if depth and j in {k for k,_ in q["slots"]} else 0.
        assert abs(sum(p.intersection(aperture).area for p,d in zip(polys,planes) if abs(d-expect)<1e-6)-aperture.area)<1e-6
        assert sum(p.intersection(aperture).area for p,d in zip(polys,planes) if abs(d-expect)>=1e-6)<1e-6

def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    cases=arch.fixtures();styles={c["building_id"]:c["id"] for c in cases}
    op=tile.plan_facades
    try:
        tile.plan_facades=arch.styled_plan
        g0,t0,cells,owners=tile.compile_scene(cases,audit_samples=2048)
    finally:tile.plan_facades=op
    joints={r["facade_id"]:corr.panel_joints(r,cells) for r in t0 if r["role"]=="wall" and styles[r["building"]]=="panel_block"}
    plans={};anchors=set()
    for c in cases:
        p,fac,fail=arch.plan_modules(c);rm,rf=roof.plan(c)
        assert not fail and not rf
        base=variants.variants(c,p,construction.improve(c,fac,rm),True)
        anchors|={m["anchor"] for m in base if m["kind"].startswith("balcony_")}
        plans[c["id"]]=(base,plan_recesses(c,p,base,t0,cells))
    assert len(anchors)==86
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    cameras=[((61,6,10),4.3,(.35,-1,.45)),((58,4,11),23.,(.5,-1,.22)),
             ((49,0,12),34.,(1.,-1,.12)),((67,12,12),34.,(-1,1,.12)),
             ((58,6,13),22.,(.4,1,.3))]
    sheet=Image.new("RGB",(1920,640*len(cameras)+30),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLB | projected / recessed 0.6 m / recessed 1.0 m",fill=(0,0,0))
    old=ref.shader;ref.shader=variants.variant_shader(old,cells,styles,joints)
    reports=[];baseline_vol=None
    try:
        for i,depth in enumerate(DEPTHS):
            dest=OUT/f"depth_{depth:.1f}";dest.mkdir(exist_ok=True)
            groups=copy.deepcopy(g0);table=copy.deepcopy(t0);placements=[]
            for c in cases:
                base,qs=plans[c["id"]];mods=base
                if depth and qs:
                    selected=set.union(*(q["selected"] for q in qs))
                    mods=[m for m in base if m["id"] not in selected]
                    for q in qs:
                        reveal,loggias=recess(groups,table,c,q,depth)
                        mods += [reveal]+loggias
                    # Recesses belong to opposite walls; their cavity
                    # volumes must not overlap. Other existing attached
                    # modules must not enter the carved region.
                    cavity=[]
                    for q in qs:
                        x,z,X,Z=q["cut"].bounds;pos,_,_,_=frame(q["a"])
                        uv=np.array([[x,z],[X,z],[X,Z],[x,Z]])
                        vv=np.concatenate([pos(uv),pos(uv,depth)])
                        cavity.append((vv.min(0),vv.max(0)))
                    for j,(lo,hi) in enumerate(cavity):
                        for l,h in cavity[:j]:assert not np.all(np.minimum(hi,h)-np.maximum(lo,l)>1e-8)
                        for m in base:
                            if m["id"] in selected:continue
                            for _,_,t in m["geometry"]:
                                v0=t.reshape(-1,3);l=v0.min(0);h=v0.max(0)
                                assert not np.all(np.minimum(hi,h)-np.maximum(lo,l)>1e-8),(m["id"],"cavity clash")
                for m in mods:
                    combined.append_compact(groups,table,c,m)
                    placements.append({k:v for k,v in m.items() if k!="geometry"})
            assert len({m["id"] for m in placements})==len(placements)
            assert {m["anchor"] for m in placements if m["kind"].startswith("balcony_") or m["kind"]=="recessed_loggia"}==anchors
            assert table[:len(t0)]==t0
            path=dest/"tile.glb";tile.export_glb(groups,path);v,uv,ids,_=tile.read_glb(path,groups,table)
            host,hosttris=host_mesh(v,ids,table,"facade:panel_block")
            if baseline_vol is None:baseline_vol=host.volume
            removed=sum(q["cut"].area*depth for q in plans["panel_block"][1])
            assert abs(baseline_vol-host.volume-removed)<1e-4
            for c in cases:
                for q in plans[c["id"]][1]:
                    audit_projection(v,uv,ids,q,cells,depth,np.asarray(c["offset"]))
            (dest/"facade_cells.bin").write_bytes(cells.tobytes())
            samples=pack_table(table,cells,dest/"facade_params.bin")
            check=json.loads(subprocess.run([decoder,str(dest/"facade_params.bin"),str(dest/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
            assert check["records"]==len(table) and check["cells"]==len(cells)
            (dest/"parameters.json").write_text(json.dumps(table,indent=2))
            (dest/"placements.json").write_text(json.dumps(placements,indent=2))
            # Same canonical wall record and UV coordinates, even after
            # displacement; no claim that opaque front masks preserve
            # the visible image of the recessed emission behind them.
            for r in t0:
                if r["role"]!="wall":continue
                s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
                a,b=ref.shader(s,r,maps,"night");x,y=ref.shader(s,table[r["id"]],maps,"night")
                np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
            for row,(center,ppm,view) in enumerate(cameras):
                im,dep=ref.render(v,uv,ids,table,maps,"day",center_override=center,ppm_override=ppm,view_override=view)
                im.save(dest/f"view_{row}.png");ImageDraw.Draw(im).text((8,8),f"depth {depth:.1f}m",fill=(0,0,0));sheet.paste(im,(i*640,row*640+30))
            im,_=ref.render(v,uv,ids,table,maps,"night",center_override=cameras[1][0],ppm_override=cameras[1][1],view_override=cameras[1][2])
            im.save(dest/"night.png")
            loggias=[m for m in placements if m["kind"]=="recessed_loggia"]
            reports.append(dict(depth_m=depth,triangles=len(v)//3,records=len(table),glb_bytes=path.stat().st_size,
                panel_host_triangles=hosttris,panel_host_volume_m3=float(host.volume),removed_volume_m3=removed,
                balcony_anchors=len(anchors),recessed_anchors=len(loggias),loggia_variants=dict(Counter(m["envelope_variant"] for m in loggias)),
                opening_triangles_added=0,abi_samples=samples,cpp_abi=True,
                cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest()))
        assert reports[0]["triangles"]==3714
        assert reports[0]["panel_host_triangles"]==12
        assert reports[1]["panel_host_triangles"]==reports[2]["panel_host_triangles"]==44
        assert reports[1]["recessed_anchors"]==reports[2]["recessed_anchors"]==12
        assert all(reports[1]["loggia_variants"].get(k,0)>0 for k in ("open","full","partial"))
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report=dict(schema="pcg.research.recessed-loggias.v1",lanes=reports,
        limitations=["Two opposing continuous recesses in one synthetic rectangular panel building; arbitrary polygons and corner loggias not validated",
        "Host is watertight; architectural slab/guard skins are not separate watertight solids, intentionally omit buried contact faces",
        "Front envelope glazing is opaque shader proxy; canonical opening emission unchanged but visible radiance can change by occlusion",
        "Existing PCGF v1 record layout; reveal/guard style remains diagnostic sidecar, no native adapter or collision acceptance",
        "CPU diffuse geometry diagnostic, no weather effects or runtime LOD"])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("RECESSED_LOGGIA_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=76,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"RECESSED_LOGGIA_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
