"""Reference-driven architectural corrections; SDK-free research only.
Existing canonical openings, compact GLB exporter, roof support and massing.
No per-window geometry or lights; dormer window is a mask on its host wall.
"""
from pathlib import Path
from collections import Counter
import base64,copy,io,json,math,sys,subprocess,hashlib
import numpy as np
import trimesh
from PIL import Image,ImageDraw
from shapely.geometry import box,Polygon
from pcg.footprint import Footprint
from pcg.massing import build_massing
from pcg.materials import MATERIALS,texture_maps
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table,LAYERS,opening_rectangles
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_facade_architecture as arch
import zero_roof_budgets as roof
import zero_combined_lod as combined

OUT=Path("artifacts/zero-architecture-correctness")
CLEAR=.055
HALF=.025

def panel_joints(r,cells):
    """Panel seams are metric rectangles, planned from actual apertures.
    Vertical seams choose centers of free inter-opening gaps per floor.
    Horizontal joints split around openings crossing the floor boundary.
    """
    L,H,h=r["length"],r["height"],r["floor_h"]
    openings=[tuple(b) for _,b in opening_rectangles(r,cells).values()]
    expanded=[box(x-CLEAR,y-CLEAR,X+CLEAR,Y+CLEAR) for x,y,X,Y in openings]
    result=[]
    def emit(x,y,X,Y):
        if X-x<=1e-8 or Y-y<=1e-8:return
        geom=box(x,y,X,Y)
        for op in expanded:geom=geom.difference(op)
        for p in getattr(geom,"geoms",[geom]):
            if p.is_empty or p.area<=1e-9:continue
            # All operands axis-aligned. Decompose any notched result
            # into rectangular strips using the actual polygon edges.
            if p.geom_type!="Polygon":continue
            xs=sorted({v[0] for v in p.exterior.coords}|
                      {v[0] for ring in p.interiors for v in ring.coords})
            ys=sorted({v[1] for v in p.exterior.coords}|
                      {v[1] for ring in p.interiors for v in ring.coords})
            for a,b in zip(xs,xs[1:]):
                for c,d in zip(ys,ys[1:]):
                    q=box(a,c,b,d)
                    if q.area>1e-9 and p.covers(q):result.append((a,c,b,d))
    for i in range(1,r["floors"]):
        z=i*h
        if z+HALF<H:emit(0,z-HALF,L,z+HALF)
    for i in range(r["floors"]):
        z=i*h;Z=min((i+1)*h,H)
        if Z-z<=2*HALF:continue
        occupied=sorted((x,X) for x,y,X,Y in openings if y<Z and Y>z)
        merged=[]
        for a,b in occupied:
            if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
            else:merged.append((a,b))
        if not merged:
            centers=np.arange(2.8,L-.10,2.8)
        else:
            centers=[(a[1]+b[0])/2 for a,b in zip(merged,merged[1:]) if b[0]-a[1]>=2*(CLEAR+HALF)]
        for x in centers:emit(x-HALF,z+HALF,x+HALF,Z-HALF)
    # Independent complete rectangle intersection, not sampled pixel masking.
    for a in result:
        assert 0<=a[0]<a[2]<=L and 0<=a[1]<a[3]<=H
        for b in openings:
            assert box(*a).intersection(box(*b).buffer(CLEAR-1e-8,join_style=2)).area<1e-10
    return sorted(set(result))

def seam_mask(rects,sample):
    u,z=sample.T;out=np.zeros(len(sample),bool)
    for x,y,X,Y in rects:out|=(u>=x)&(u<=X)&(z>=y)&(z<=Y)
    return out

def test_joint_regression():
    # A door crossing a floor line and an off-grid window crossing x=2.8.
    r=dict(length=8.,height=6.,floor_h=3.,floors=2,bays=2,first_center=2.8,
           spacing=2.8,window_width=1.4,window_height=1.4,sill=.9,
           balcony_threshold=.17,door_height=2.2,door=[.35,0.,1.,3.15],cell_start=0)
    cells=np.array([1,1,1,1],np.uint8)
    j=panel_joints(r,cells)
    assert j and j==panel_joints(r,cells)
    assert not seam_mask(j,np.array([[2.8,1.5],[.8,3.],[2.8,4.5]])).any()
    assert seam_mask(j,np.array([[4.2,1.5],[4.2,3.]])).all()

def mod(c,key,kind,parts):
    return dict(id=c["building_id"]+":detail:"+key,kind=kind,
                triangles=sum(len(t) for _,_,t in parts),geometry=parts)

def bm(x,y,w,d,h,z):
    return roof.box_mesh(x,y,w,d,h,z)

def project_bounds(m):
    v=np.concatenate([t.reshape(-1,3) for _,_,t in m["geometry"]])
    return box(*np.r_[v[:,:2].min(0),v[:,:2].max(0)])

def overlaps(a,b):
    A=combined.bounds(a);B=combined.bounds(b)
    return np.all(np.minimum(A[1],B[1])-np.maximum(A[0],B[0])>1e-8)

def fit_drains_to_cornice(c,fac):
    """Resolve a real clash by ending the vertical downspout at the
    underside of the new cornice. Keep the same placement and base.
    This is an attachment fit, not a collision whitelist.
    """
    limit=c["eave_m"]-.18
    result=[]
    for m in fac:
        if m["kind"]!="drain" or combined.bounds(m)[1,2]<=limit:
            result.append(m);continue
        q=copy.deepcopy(m)
        lo,hi=combined.bounds(m)
        assert lo[2]<limit<hi[2]
        for _,_,t in q["geometry"]:
            t[:,:,2]=lo[2]+(t[:,:,2]-lo[2])*(limit-lo[2])/(hi[2]-lo[2])
            mesh=trimesh.Trimesh(vertices=t.reshape(-1,3),faces=np.arange(t.size//3).reshape(-1,3),process=True)
            assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
        new=combined.bounds(q)
        np.testing.assert_allclose(new[0],lo,atol=1e-9,rtol=0)
        np.testing.assert_allclose(new[1,:2],hi[:2],atol=1e-9,rtol=0)
        assert abs(new[1,2]-limit)<1e-9
        q["bounds"]=new.tolist()
        q["fit_adjustment"]={"reason":"cornice_underside_attachment","original_top_m":float(hi[2]),"new_top_m":float(new[1,2])}
        result.append(q)
    return result

def test_drain_cornice_regression():
    # Exact failed brick_stack placement from CI artifact 15102650906.
    c={"eave_m":15.}
    m={"id":"facade:brick_stack:facade:0:0:drain","kind":"drain",
       "geometry":[("drain","painted_rail",bm(.28,-.065,.10,.13,14.8,.10))]}
    corn={"geometry":[("cornice","painted_trim",bm(15.,-.12,30.64,.40,.18,14.82))]}
    assert overlaps(m,corn)
    q=fit_drains_to_cornice(c,[m])[0]
    assert q["id"]==m["id"] and not overlaps(q,corn)
    assert abs(combined.bounds(q)[1,2]-14.82)<1e-9
    assert abs(combined.bounds(m)[1,2]-14.9)<1e-9

def roof_details(c,old,fac,spacing):
    f=Footprint(c["building_id"],c["outer"],c["holes"])
    mass=build_massing(f,c["eave_m"],c["roof"],c["rise"])
    rt=roof.triangles(mass.shell);poly=f.polygon()
    x0,y0,x1,y1=poly.bounds;w=x1-x0;d=y1-y0;h=c["eave_m"]
    flat=c["roof"]=="flat"
    kept=[combined.roof_module(m) for m in old if m["kind"]!="parapet" and
          not (not flat and m["id"].endswith(":service_housing"))]
    added=[]
    # Closed solid strips with exact corner contacts, not four disconnected boxes.
    if flat:
        t=.20;hh=.48
        pp=[bm(w/2,t/2,w,t,hh,h-.015),bm(w/2,d-t/2,w,t,hh,h-.015),
            bm(t/2,d/2,t,d-2*t,hh,h-.015),bm(w-t/2,d/2,t,d-2*t,hh,h-.015)]
        ring=mod(c,"parapet","continuous_parapet",[("parapet","concrete_panel",p) for p in pp])
        for p in pp:
            a=p.reshape(-1,3);assert poly.covers(box(*np.r_[a[:,:2].min(0),a[:,:2].max(0)]))
        assert abs(sum((p.reshape(-1,3).max(0)-p.reshape(-1,3).min(0))[:2].prod() for p in pp)-(w*d-(w-2*t)*(d-2*t)))<1e-7
        for p in pp:
            q=mod(c,"test","parapet",[("parapet","concrete_panel",p)])
            assert not any(overlaps(q,b) for b in kept)
        added.append(ring)
    # Eave/cornice: continuous along flat perimeter, on gable eaves
    # only along long sides. Deliberate attachment overlap at the host wall.
    corn=[bm(w/2,-.12,w+.64,.40,.18,h-.18),
          bm(w/2,d+.12,w+.64,.40,.18,h-.18)]
    if flat:
        corn.extend([bm(-.12,d/2,.40,d-.16,.18,h-.18),
                     bm(w+.12,d/2,.40,d-.16,.18,h-.18)])
    for p in corn:
        q=mod(c,"test","cornice",[("cornice","painted_trim",p)])
        clashes=[b["id"] for b in fac if overlaps(q,b)]
        assert not clashes,(c["id"],"cornice clash",clashes)
    added.append(mod(c,"cornice","cornice",[("cornice","painted_trim",p) for p in corn]))
    # Real open metal railing, rectangular cross-sections. No alpha-test
    # or billboard. Near/far support count changes, not the rail silhouette.
    edges=[((.10,.10),(w-.10,.10)),((.10,d-.10),(w-.10,d-.10))] if flat else [
           ((.18,.20),(w-.18,.20)),((.18,d-.20),(w-.18,d-.20))]
    if flat:edges.extend([((.10,.10),(.10,d-.10)),((w-.10,.10),(w-.10,d-.10))])
    posts={};rails=[]
    for k,(a,b) in enumerate(edges):
        a=np.array(a);b=np.array(b);length=np.linalg.norm(b-a)
        n=int(math.ceil(length/spacing));horizontal=abs(a[1]-b[1])<1e-8
        for p in np.linspace(a,b,n+1):
            x,y=p;key=tuple(np.round(p,7))
            if key in posts:continue
            if flat:base=h+.465
            else:
                support=[roof.roof_z(rt,x+dx,y+dy) for dx in (-.018,.018) for dy in (-.018,.018)]
                base=min(support)-.015
            posts[key]=bm(x,y,.036,.036,.86,base)
        mid=(a+b)/2
        z=h+.465 if flat else roof.roof_z(rt,*mid)-.015
        # Constant height on each eave-parallel segment.
        for height in (.44,.82):
            rails.append(bm(*mid,length if horizontal else .035,.035 if horizontal else length,.035,z+height))
    guard=mod(c,"guard","roof_guard",[("guard","painted_rail",p) for p in list(posts.values())+rails])
    guard.update(posts=len(posts),rail_segments=len(rails),spacing_m=spacing)
    for _,_,p in guard["geometry"]:
        q=mod(c,"test","guard",[("guard","painted_rail",p)])
        assert not any(overlaps(q,b) for b in kept)
    added.append(guard)
    if not flat:
        for i,cy in enumerate((.22*d,.78*d)):
            cx=.48*w;ww=1.5;dd=2.1
            foot=Footprint(c["building_id"]+":dormer:"+str(i),
                [(cx-ww/2,cy-dd/2),(cx+ww/2,cy-dd/2),(cx+ww/2,cy+dd/2),(cx-ww/2,cy+dd/2)])
            assert poly.buffer(-.04).covers(foot.polygon())
            assert all(foot.polygon().intersection(project_bounds(q)).area<1e-9 for q in kept)
            z=[roof.roof_z(rt,x,y) for x,y in foot.exterior]
            bottom=min(z)-.035;top=max(z)+.72
            # Long axis points downhill: front gable and two roof slopes.
            s=build_massing(foot,top-bottom,"gable",.48).shell
            v=np.asarray(s.vertices)+[0,0,bottom]
            tri=np.array([t for _,t in s.triangles()])
            mesh=trimesh.Trimesh(vertices=v,faces=tri,process=False)
            assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
            parts=[]
            for surface in s.surfaces:
                ts=np.array([v[list(t)] for sid,t in s.triangles() if sid==surface.id])
                if surface.role=="roof":kind,mat="dormer_roof","metal_roof_red"
                elif surface.role=="wall":
                    n=combined.normal(ts[0])
                    front=n[1]<-.99 if i==0 else n[1]>.99
                    kind,mat=("dormer_front" if front else "dormer_wall"),"plaster_cream"
                else:kind,mat="dormer_base","concrete_panel"
                parts.append((kind,mat,ts))
            assert sum(k=="dormer_front" for k,_,_ in parts)==1
            q=mod(c,"dormer"+str(i),"gable_dormer",parts)
            q.update(support_m=[min(z),max(z)],attic_opening="shader_only")
            assert combined.bounds(q)[0,2]<min(z) and combined.bounds(q)[1,2]>max(z)
            added.append(q)
    return kept+added

def shader(old,cells,styles,joints,enhanced,details=False):
    base=arch.render_shader(old,cells,styles,True,
        (lambda r,s:seam_mask(joints.get(r["facade_id"],[]),s)) if enhanced else None)
    def shade(sample,r,maps,mode,dense=False):
        col,em=base(sample,r,maps,mode,dense)
        k=r.get("accessory","")
        if details and k=="dormer_front":
            x,y=sample.T/np.asarray(r["span"])[:,None]
            mask=(x>.22)&(x<.78)&(y>.46)&(y<.69)
            glass=mask&(x>.27)&(x<.73)&(y>.49)&(y<.66)&(np.abs(x-.5)>.035)
            col[mask]=srgb_decode(np.array([.78,.76,.70]))
            col[glass]=srgb_decode(np.array([.19,.25,.29]))
        if details and k in ("rail","glazed"):
            x,y=sample.T/np.asarray(r["span"])[:,None]
            v=ref.seed(r["placement"])%5
            palette=np.array([[.33,.22,.17],[.46,.13,.11],[.66,.66,.62],[.25,.27,.29],[.44,.35,.27]])
            mask=(y<=.39) if k=="glazed" else np.ones(len(x),bool)
            col[mask]=srgb_decode(palette[v])
            grooves=(np.mod((x if v!=2 else y)*24,1)<.15)
            col[mask&grooves]*=.72
            if k=="glazed":
                bars=3+v%3
                glass=y>.39
                frames=glass&((np.mod(x*bars,1)<.075)|(x>.97)|(y>.96)|(np.abs(y-.41)<.025))
                col[glass]=srgb_decode(np.array([.22,.33,.37])*(.85+.065*v))
                col[frames]=srgb_decode(np.array([.79,.77,.72]) if v!=4 else np.array([.37,.23,.14]))
        return col,em
    return shade

def main(decoder):
    test_joint_regression()
    test_drain_cornice_regression()
    OUT.mkdir(parents=True,exist_ok=True);cases=arch.fixtures()
    styles={c["building_id"]:c["id"] for c in cases}
    oldplan=tile.plan_facades
    try:
        tile.plan_facades=arch.styled_plan
        g0,t0,cells,owners=tile.compile_scene(cases,audit_samples=2048)
    finally:tile.plan_facades=oldplan
    joints={r["facade_id"]:panel_joints(r,cells) for r in t0 if r["role"]=="wall" and styles[r["building"]]=="panel_block"}
    assert sum(map(len,joints.values()))>0
    (OUT/"panel_joints.json").write_text(json.dumps(joints,indent=2))
    plans={};anchors=set()
    for c in cases:
        p,fac,fail=arch.plan_modules(c);rm,rf=roof.plan(c)
        assert not fail and not rf
        want={o.id for a in p.facades for o in a.openings if o.role=="balcony_door"}
        assert want=={m["anchor"] for m in fac if m["kind"].startswith("balcony_")}
        anchors|=want
        plans[c["id"]]=(fac,rm)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    lanes=("baseline","corrected_joints","roof_economy","roof_detailed")
    cameras=[((61,6,10),4.3,(.35,-1,.45)),
             ((58,2,13),24.,(.01,-1,.06)),
             ((58,6,21),30.,(.25,-1,.42)),
             ((107,3,16),34.,(.25,-1,.50)),
             ((107,8,10),19.,(.30,1,.28)),
             ((15,0,10),18.,(.30,-1,.18))]
    sheet=Image.new("RGB",(640*len(lanes),640*len(cameras)+32),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLB | baseline / aperture-aware joints / roof economy / roof detailed",fill=(0,0,0))
    reports=[];old=ref.shader;first=None
    try:
        for col,lane in enumerate(lanes):
            dest=OUT/lane;dest.mkdir(exist_ok=True)
            groups=copy.deepcopy(g0);table=copy.deepcopy(t0);placements=[]
            for c in cases:
                fac,rm=plans[c["id"]]
                if col<2:mods=fac+[combined.roof_module(m) for m in rm]
                else:
                    fitted=fit_drains_to_cornice(c,fac)
                    mods=fitted+roof_details(c,rm,fitted,2.4 if col==2 else 1.2)
                for m in mods:
                    combined.append_compact(groups,table,c,m)
                    placements.append({k:v for k,v in m.items() if k!="geometry"})
            assert table[:len(t0)]==t0
            assert {m["anchor"] for m in placements if m["kind"].startswith("balcony_")}==anchors
            assert len({r["surface"] for r in table})==len(table)
            for role in tile.GROUPS:
                for k,a in g0[role].items():np.testing.assert_array_equal(groups[role][k][:len(a)],a)
            path=dest/"tile.glb";tile.export_glb(groups,path);v,uv,ids,_=tile.read_glb(path,groups,table)
            assert len(v)//3==sum(len(a["f"]) for a in g0.values())+sum(m["triangles"] for m in placements)
            if col==0:first=(v.copy(),uv.copy(),ids.copy())
            if col==1:
                for a,b in zip(first,(v,uv,ids)):np.testing.assert_array_equal(a,b)
            (dest/"facade_cells.bin").write_bytes(cells.tobytes())
            samples=pack_table(table,cells,dest/"facade_params.bin")
            check=json.loads(subprocess.run([decoder,str(dest/"facade_params.bin"),str(dest/"facade_cells.bin"),str(len(LAYERS))],check=True,capture_output=True,text=True).stdout)
            assert check["records"]==len(table) and check["cells"]==len(cells)
            (dest/"parameters.json").write_text(json.dumps(table,indent=2))
            (dest/"placements.json").write_text(json.dumps(placements,indent=2))
            ref.shader=shader(old,cells,styles,joints,col>0,col>1)
            for r in t0:
                if r["role"]!="wall":continue
                sample=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
                a,b=shader(old,cells,styles,joints,False)(sample,r,maps,"night")
                c,e=ref.shader(sample,r,maps,"night")
                np.testing.assert_array_equal(b,e)
                role,_,_=tile.lookup(sample,r,cells)
                np.testing.assert_array_equal(a[role!=0],c[role!=0])
            for row,(center,ppm,view) in enumerate(cameras):
                im,dep=ref.render(v,uv,ids,table,maps,"day",center_override=center,ppm_override=ppm,view_override=view)
                im.save(dest/f"view_{row}.png");ImageDraw.Draw(im).text((8,8),lane,fill=(0,0,0));sheet.paste(im,(col*640,row*640+32))
            im,_=ref.render(v,uv,ids,table,maps,"night",center_override=cameras[0][0],ppm_override=cameras[0][1],view_override=cameras[0][2]);im.save(dest/"night.png")
            reports.append(dict(lane=lane,triangles=len(v)//3,records=len(table),glb_bytes=path.stat().st_size,
                balcony_anchors=len(anchors),opening_triangles_added=0,abi_samples=samples,cpp_abi=True,
                modules=dict(Counter(m["kind"] for m in placements)),
                drain_fit_adjustments=[m["fit_adjustment"] for m in placements if "fit_adjustment" in m],
                guard_posts=sum(m.get("posts",0) for m in placements),
                guard_rails=sum(m.get("rail_segments",0) for m in placements),
                cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest()))
        assert reports[0]["triangles"]==reports[1]["triangles"]
        assert reports[2]["modules"].get("gable_dormer")==2
        assert reports[2]["modules"].get("continuous_parapet")==2
        assert reports[2]["modules"].get("roof_guard")==3
        assert reports[3]["guard_posts"]>reports[2]["guard_posts"]>0
        assert reports[3]["triangles"]>reports[2]["triangles"]
        assert len(reports[2]["drain_fit_adjustments"])==len(reports[3]["drain_fit_adjustments"])==1
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report=dict(schema="pcg.research.architecture-correctness.v1",lanes=reports,
        panel_joint_rectangles=sum(map(len,joints.values())),opening_clearance_m=CLEAR,
        limitations=["rectangular fixtures only; gable dormers on one existing gable archetype",
        "dormer attic mask is dark, not part of canonical occupancy ABI; no window geometry",
        "balcony finish variants only; true recessed loggias, corners and partial-glazed geometry deferred",
        "roof rails and parapets use closed boxes, not Boolean union; intentional attachment contacts",
        "cornice strips approximate eave overhang, not replacement roof surface",
        "CPU diffuse diagnostics only, no native or hardware GPU timing; no weather effects"])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("ARCH_CORRECT_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1400,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=77,optimize=True)
    s=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(s),4000):print(f"ARCH_CORRECT_PREVIEW_{i//4000:03d} "+s[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
