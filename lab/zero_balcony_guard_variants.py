from pathlib import Path
from collections import Counter
import base64,copy,hashlib,io,json,sys,subprocess
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from pcg.footprint import Footprint
from pcg.massing import build_massing
from pcg.materials import MATERIALS,texture_maps
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table,LAYERS
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_facade_architecture as arch
import zero_roof_budgets as roof
import zero_combined_lod as combined
import zero_architecture_correctness as corr


"""Stage 15: envelope variation and closed low-cost roof guard sections.
No window geometry; only architectural envelope surfaces, as in stage 14.
"""
import zero_roof_construction as construction
from pcg.architecture import BALCONY_DEPTH
from shapely.geometry import MultiPoint

def prism8(tris):
    # A closed triangular-section prism along the dominant box axis.
    # Preserve the measured outer AABB; this is a geometric approximation,
    # not a claim of identical cross-section, stiffness or shadow.
    lo,hi=tris.reshape(-1,3).min(0),tris.reshape(-1,3).max(0)
    axis=int(np.argmax(hi-lo));a,b=[i for i in range(3) if i!=axis]
    v=[]
    for end in (lo[axis],hi[axis]):
        for x,y in ((lo[a],lo[b]),(hi[a],lo[b]),((lo[a]+hi[a])/2,hi[b])):
            p=np.zeros(3);p[axis]=end;p[a]=x;p[b]=y;v.append(p)
    faces=np.array([(0,2,1),(3,4,5),(0,1,4),(0,4,3),(1,2,5),(1,5,4),(2,0,3),(2,3,5)])
    mesh=trimesh.Trimesh(vertices=v,faces=faces,process=False)
    if mesh.volume<0:faces=faces[:,::-1]
    mesh=construction.solid(v,faces)
    construction.require_connected_faces(mesh)
    np.testing.assert_allclose(mesh.bounds,[lo,hi],atol=1e-9,rtol=0)
    assert len(mesh.faces)==8
    return mesh.vertices[mesh.faces]

def guard_lod(m):
    if m["kind"]!="roof_guard":return m
    parts=[(k,f,prism8(t)) for k,f,t in m["geometry"]]
    q=dict(m,geometry=parts,triangles=sum(len(t) for _,_,t in parts))
    assert q["triangles"]*3==m["triangles"]*2
    np.testing.assert_allclose(combined.bounds(q),combined.bounds(m),atol=1e-9,rtol=0)
    return q

def quad(a,b,c,d):
    return np.array([[a,b,c],[a,c,d]])

def section(t,low,high,left=0.,right=1.):
    # Existing arch.panel_quad contract: bottom-left, bottom-right,
    # top-right and top-left. Restrict its ruled rectangle in metric space.
    a,b,c,d=t[0,0],t[0,1],t[0,2],t[1,2]
    np.testing.assert_allclose(t[1,:2],[a,c],atol=1e-9,rtol=0)
    def p(x,y):return (1-y)*((1-x)*a+x*b)+y*((1-x)*d+x*c)
    return quad(p(left,low),p(right,low),p(right,high),p(left,high))

def partial(m):
    # Three side skins ordered left, front, right by the existing recipe.
    # Half of the original envelope stays full-height; the other half
    # becomes an open balcony. No separate glazing/frame meshes.
    parts=m["geometry"]
    assert [k for k,_,_ in parts]==["slab","glazed","glazed","glazed","slab"]
    frac=.38
    new=[parts[0]]
    for j,(k,f,t) in enumerate(parts[1:4]):
        new.append(("partial_lower",f,section(t,0.,frac)))
        if j==0:new.append(("partial_upper",f,section(t,frac,1.)))
        if j==1:
            new.append(("partial_upper",f,section(t,frac,1.,0.,.5)))
    _,f,t=parts[-1]
    a,b,c,d=t[0,0],t[0,1],t[0,2],t[1,2]
    # Original cap order is front-left, front-right, rear-right, rear-left.
    new.append(("slab",f,quad(a,(a+b)/2,(d+c)/2,d)))
    return new

def balcony_variant(m,facades):
    if not m["kind"].startswith("balcony_"):return m
    q=copy.deepcopy(m);a=facades[m["facade"]];seed=ref.seed(m["id"])
    family="full" if m["kind"]=="balcony_glazed" else "open"
    if family=="full" and seed%3==0:
        q["geometry"]=partial(q);family="partial"
    # Depth variation stays inside the original collision-tested envelope.
    # Slab width, host attachment and canonical balcony-door remain unchanged.
    factor=.80 if seed%4==0 else 1.
    origin=np.asarray(a.start);normal=np.asarray(a.outward)
    for k,f,t in q["geometry"]:
        p=t[:,:,:2];dist=(p-origin)@normal
        p-=dist[:,:,None]*normal*(1-factor)
    old=combined.bounds(m);new=combined.bounds(q)
    assert np.all(new[0]>=old[0]-1e-8) and np.all(new[1]<=old[1]+1e-8)
    np.testing.assert_allclose(new[:,2],old[:,2],rtol=0,atol=1e-8)
    vertices=np.concatenate([t.reshape(-1,3) for _,_,t in q["geometry"]])
    dist=(vertices[:,:2]-origin)@normal
    assert abs(dist.min())<1e-8 and abs(dist.max()-BALCONY_DEPTH*factor)<1e-8
    for _,_,t in q["geometry"]:
        assert np.all(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)>1e-8)
    q.update(triangles=sum(len(t) for _,_,t in q["geometry"]),bounds=new.tolist(),
             envelope_variant=family,depth_m=BALCONY_DEPTH*factor)
    return q

def variants(c,p,baseline,economy):
    facades={a.id:a for a in p.facades}
    out=[balcony_variant(m,facades) for m in baseline]
    if economy:out=[guard_lod(m) for m in out]
    assert [m["id"] for m in out]==[m["id"] for m in baseline]
    for m in out:
        if not m["kind"].startswith("balcony_"):continue
        for other in out:
            if other["id"]==m["id"] or other["kind"] in ("continuous_parapet","cornice","roof_guard","pitched_overhang","open_gutters"):continue
            assert not corr.overlaps(m,other),(m["id"],other["id"])
    return out

def variant_shader(old,cells,styles,joints):
    base=construction.shader(old,cells,styles,joints,True)
    def shade(s,r,maps,mode,dense=False):
        k=r.get("accessory")
        if k=="partial_lower":
            return base(s,dict(r,accessory="rail"),maps,mode,dense)
        col,em=base(s,r,maps,mode,dense)
        if k=="partial_upper":
            x,y=s.T/np.asarray(r["span"])[:,None]
            bars=2+ref.seed(r["placement"])%3
            frame=(np.mod(x*bars,1)<.075)|(x>.96)|(y<.045)|(y>.955)
            col[:]=srgb_decode(np.array([.22,.33,.37]))
            col[frame]=srgb_decode(np.array([.79,.77,.73]))
        return col,em
    return shade

def test_guard():
    for dims in ((.036,.036,.86),(2.4,.035,.035),(.035,2.4,.035)):
        t=corr.bm(0.,0.,*dims,0.)
        q=prism8(t)
        assert q.shape==(8,3,3)


OUT=Path("artifacts/zero-balcony-guard-variants")

def main(decoder):
    test_guard()
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
        anchors|={o.id for a in p.facades for o in a.openings if o.role=="balcony_door"}
        fitted=corr.fit_drains_to_cornice(c,fac)
        baseline=construction.improve(c,fac,rm)
        candidate=variants(c,p,baseline,False)
        economy=variants(c,p,baseline,True)
        assert {m["id"] for m in baseline}<={m["id"] for m in candidate}
        plans[c["id"]]=(baseline,candidate,economy)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    cameras=[((61,6,10),4.3,(.35,-1,.45)),((107,0,9),32.,(.25,-1,.16)),((107,12,9),32.,(.25,1,.16)),((58,5,21),28.,(.25,-1,.42)),((107,6,16),24.,(.65,-1,1.8))]
    sheet=Image.new("RGB",(1920,640*len(cameras)+30),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLB | stage 14 / balcony variants / compact guard prisms",fill=(0,0,0))
    old=ref.shader;reports=[];before=None
    try:
        for i,name in enumerate(("baseline","variants","economy")):
            d=OUT/name;d.mkdir(exist_ok=True)
            g=copy.deepcopy(g0);table=copy.deepcopy(t0);placements=[]
            for c in cases:
                for m in plans[c["id"]][i]:
                    combined.append_compact(g,table,c,m)
                    placements.append({k:v for k,v in m.items() if k!="geometry"})
            assert len({m["id"] for m in placements})==len(placements)
            assert {m["anchor"] for m in placements if m["kind"].startswith("balcony_")}==anchors
            assert table[:len(t0)]==t0
            for role in tile.GROUPS:
                for k,a in g0[role].items():np.testing.assert_array_equal(g[role][k][:len(a)],a)
            path=d/"tile.glb";tile.export_glb(g,path);v,uv,ids,_=tile.read_glb(path,g,table)
            assert len(v)//3==sum(len(a["f"]) for a in g0.values())+sum(m["triangles"] for m in placements)
            (d/"facade_cells.bin").write_bytes(cells.tobytes())
            samples=pack_table(table,cells,d/"facade_params.bin")
            check=json.loads(subprocess.run([decoder,str(d/"facade_params.bin"),str(d/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
            assert check["records"]==len(table) and check["cells"]==len(cells)
            (d/"parameters.json").write_text(json.dumps(table,indent=2))
            (d/"placements.json").write_text(json.dumps(placements,indent=2))
            ref.shader=variant_shader(old,cells,styles,joints) if i else construction.shader(old,cells,styles,joints,True)
            for r in t0:
                if r["role"]!="wall":continue
                s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
                a,b=construction.shader(old,cells,styles,joints,True)(s,r,maps,"night")
                x,y=ref.shader(s,r,maps,"night")
                np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
            depths=[];images=[]
            for row,(center,ppm,view) in enumerate(cameras):
                im,dep=ref.render(v,uv,ids,table,maps,"day",center_override=center,ppm_override=ppm,view_override=view)
                im.save(d/f"view_{row}.png");images.append(np.asarray(im).copy());depths.append(dep)
                ImageDraw.Draw(im).text((8,8),name,fill=(0,0,0));sheet.paste(im,(i*640,row*640+30))
            delta=[]
            if before is None:before=(images,depths)
            else:
                for a,b,da,db in zip(before[0],images,before[1],depths):
                    both=np.isfinite(da)&np.isfinite(db)
                    delta.append(dict(changed_rgb_pixels=int(np.any(a!=b,axis=2).sum()),
                        silhouette_xor_pixels=int(np.sum(np.isfinite(da)!=np.isfinite(db))),
                        common_depth_changed=int(np.sum(np.abs(da[both]-db[both])>1e-5))))
                assert sum(q["changed_rgb_pixels"] for q in delta)>100
                assert sum(q["silhouette_xor_pixels"] for q in delta)>0
            reports.append(dict(lane=name,triangles=len(v)//3,records=len(table),glb_bytes=path.stat().st_size,
                modules=dict(Counter(m["kind"] for m in placements)),balcony_anchors=len(anchors),
                envelope_variants=dict(Counter(m["envelope_variant"] for m in placements if "envelope_variant" in m)),
                shallow_balconies=sum(m.get("depth_m",BALCONY_DEPTH)<BALCONY_DEPTH for m in placements),
                guard_triangles=sum(m["triangles"] for m in placements if m["kind"]=="roof_guard"),
                opening_triangles_added=0,abi_samples=samples,cpp_abi=True,
                cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest(),view_delta=delta))
        assert reports[0]["triangles"]==4114
        assert all(reports[1]["envelope_variants"].get(k,0)>0 for k in ("open","full","partial"))
        assert reports[1]["shallow_balconies"]>0
        assert reports[1]["envelope_variants"]==reports[2]["envelope_variants"]
        assert reports[2]["guard_triangles"]*3==reports[1]["guard_triangles"]*2
        assert reports[1]["triangles"]-reports[2]["triangles"]==reports[1]["guard_triangles"]-reports[2]["guard_triangles"]
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report=dict(schema="pcg.research.balcony-guard-variants.v1",lanes=reports,
        limitations=["synthetic rectangular fixtures, CPU diffuse, not native PBR",
        "partial balcony is an architectural envelope proxy, opaque glazing illusion",
        "triangular guard cross-section changes silhouette, no structural/collision equivalence",
        "no true recessed loggia, corners, or runtime LOD; no weather effects"])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("BALCONY_GUARD_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=76,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"BALCONY_GUARD_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
