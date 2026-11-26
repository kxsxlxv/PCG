"""Research 14: roof construction detail on the same canonical GLB.
Retain stage 13 as a paired baseline. Use existing shell/roof/box/ABI code.
Only architectural silhouette volumes, not geometric window or door inserts.
"""
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

OUT=Path("artifacts/zero-roof-construction")
OVER=.32
THICK=.075

def solid(v,f):
    m=trimesh.Trimesh(vertices=np.asarray(v,float),faces=np.asarray(f,int),process=False)
    assert m.is_watertight and m.is_winding_consistent and m.volume>1e-10
    assert np.all(m.area_faces>1e-10)
    return m

def slab(rect,height,thickness):
    """Prismatic extension of an existing planar slope, no roof regeneration."""
    x,y,X,Y=rect
    xy=[(x,y),(X,y),(X,Y),(x,Y)]
    top=np.array([(a,b,height(a,b)) for a,b in xy])
    v=np.concatenate([top-[0,0,thickness],top])
    faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
    for i in range(4):
        j=(i+1)%4;faces.extend([(i,j,j+4),(i,j+4,i+4)])
    m=solid(v,faces)
    np.testing.assert_allclose(m.volume,(X-x)*(Y-y)*thickness,rtol=0,atol=1e-8)
    return [("eave_cover","ceramic_roof",m.vertices[m.faces[2:4]]),
            ("soffit","painted_trim",m.vertices[m.faces[[0,1,*range(4,12)]]])]

def overhang(c):
    w=c["outer"][1][0];d=c["outer"][2][1];h=c["eave_m"];rise=c["rise"]
    assert c["roof"]=="gable" and w>d and not c["holes"]
    f=Footprint(c["building_id"],c["outer"],c["holes"])
    rt=roof.triangles(build_massing(f,h,"gable",rise).shell)
    def height(x,y):return h+rise*min(y,d-y)/(d/2)
    # Independent agreement with the actual canonical roof planes.
    for t in rt:
        for x,y,z in t:assert abs(height(x,y)-z)<1e-8
    for x in (0.,w/2,w):
        for y in (0.,d/4,d/2,3*d/4,d):
            assert abs(height(x,y)-roof.roof_z(rt,x,y))<1e-8
    rects=[(-OVER,-OVER,w+OVER,0.),(-OVER,d,w+OVER,d+OVER),
           (-OVER,0.,0.,d/2),(-OVER,d/2,0.,d),
           (w,0.,w+OVER,d/2),(w,d/2,w+OVER,d)]
    polys=[box(*r) for r in rects];target=box(-OVER,-OVER,w+OVER,d+OVER).difference(f.polygon())
    assert unary_union(polys).symmetric_difference(target).area<1e-8
    assert abs(sum(p.area for p in polys)-target.area)<1e-8
    assert all(p.intersection(f.polygon()).area<1e-9 for p in polys)
    parts=[q for r in rects for q in slab(r,height,THICK)]
    m=corr.mod(c,"pitched_overhang","pitched_overhang",parts)
    m["projection_area_m2"]=target.area
    m["extension_m"]=OVER
    return m

def coping(c,m):
    """Replace the top 45 mm of parapet body with a cap, preserving guard seat."""
    w=c["outer"][1][0];d=c["outer"][2][1];h=c["eave_m"]
    q=copy.deepcopy(m);cap=.045;top=h+.465;t=.20;a=.025
    parts=[]
    for k,f,t0 in q["geometry"]:
        lo=t0.reshape(-1,3).min(0);hi=t0.reshape(-1,3).max(0)
        assert abs(hi[2]-top)<1e-8
        parts.append((k,f,corr.bm((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,
            hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]-cap,lo[2])))
    rects=[(-a,-a,w+a,t+a),(-a,d-t-a,w+a,d+a),
           (-a,t+a,t+a,d-t-a),(w-t-a,t+a,w+a,d-t-a)]
    polys=[box(*r) for r in rects]
    target=box(-a,-a,w+a,d+a).difference(box(t+a,t+a,w-t-a,d-t-a))
    assert unary_union(polys).symmetric_difference(target).area<1e-8
    assert abs(sum(p.area for p in polys)-target.area)<1e-8
    for x,y,X,Y in rects:
        parts.append(("coping","painted_trim",corr.bm((x+X)/2,(y+Y)/2,X-x,Y-y,cap,top-cap)))
    q.update(geometry=parts,triangles=sum(len(t0) for _,_,t0 in parts),cap_m=cap)
    assert abs(combined.bounds(q)[1,2]-top)<1e-8
    return q

def access_cap(c,m):
    # Reuse actual access housing, separate its existing front wall for
    # a metric door mask; add only a thin projecting cover, no door mesh.
    q=copy.deepcopy(m);lo,hi=combined.bounds(m);parts=[]
    for k,f,t in q["geometry"]:
        front=np.array([combined.normal(a)[1]<-.999 for a in t])
        if front.any():parts.append(("access_front",f,t[front]))
        if (~front).any():parts.append((k,f,t[~front]))
    assert sum(k=="access_front" for k,_,_ in parts)==1
    parts.append(("access_cover","bitumen_roof",corr.bm((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,
        hi[0]-lo[0]+.24,hi[1]-lo[1]+.24,.105,hi[2])))
    q.update(geometry=parts,triangles=sum(len(t) for _,_,t in parts))
    return q

def gutters(c):
    w=c["outer"][1][0];d=c["outer"][2][1];h=c["eave_m"]
    parts=[]
    # Open troughs, not opaque rectangular gutter blocks. Three existing
    # audited box sections per long edge. No duplicate positive-volume parts.
    for y in (-.40,d+.40):
        L=w+2*OVER
        parts.append(("gutter_floor","painted_rail",corr.bm(w/2,y,L,.16,.02,h-.26)))
        parts.extend(("gutter_side","painted_rail",corr.bm(w/2,y+dy,L,.02,.11,h-.24))
                     for dy in (-.07,.07))
    m=corr.mod(c,"gutters","open_gutters",parts)
    m["trough_bottom_m"]=h-.26
    return m

def swept_pipe(path,radius=.05,n=6):
    """Small explicit convex-section sweep. Existing box recipe cannot form
    a continuous elbow. Ring frames use the bisector of adjacent segments.
    The path is planar YZ and has no reversals. Trimesh validates topology.
    """
    p=np.asarray(path,float);delta=np.diff(p,axis=0)
    lengths=np.linalg.norm(delta,axis=1);assert np.all(lengths>2*radius)
    t=delta/lengths[:,None];assert np.max(np.abs(t[:,0]))<1e-9
    rings=[]
    for i,a in enumerate(p):
        direction=t[0] if i==0 else t[-1] if i==len(p)-1 else t[i-1]+t[i]
        direction/=np.linalg.norm(direction)
        cross=np.cross(direction,[1.,0.,0.])
        miter=1. if i in (0,len(p)-1) else 1./np.dot(direction,t[i-1])
        assert miter<1.5
        rings.extend(a+radius*(np.cos(j*2*np.pi/n)*np.array([1.,0.,0.])+
                    np.sin(j*2*np.pi/n)*miter*cross) for j in range(n))
    faces=[]
    for i in range(len(p)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n
            faces.extend([(a,b,b+n),(a,b+n,a+n)])
    for j in range(1,n-1):
        faces.extend([(0,j+1,j),((len(p)-1)*n,(len(p)-1)*n+j,(len(p)-1)*n+j+1)])
    m=solid(rings,faces)
    # Every sample has the expected cross-section and connected topology.
    assert len(m.split(only_watertight=True))==1
    return m.vertices[m.faces]

def bend_drain(c,m):
    lo,hi=combined.bounds(m);x=(lo[0]+hi[0])/2;h=c["eave_m"]
    y=(lo[1]+hi[1])/2
    path=[(x,y,lo[2]),(x,y,h-.76),(x,-.40,h-.39),(x,-.40,h-.26)]
    t=swept_pipe(path)
    q=dict(m,geometry=[("bent_drain","painted_rail",t)],triangles=len(t),
           attachment="south_gutter_floor",centerline_m=path)
    q["bounds"]=combined.bounds(q).tolist()
    assert abs(combined.bounds(q)[1,2]-(h-.26))<1e-8
    assert abs(combined.bounds(q)[0,2]-lo[2])<1e-8
    return q

def improve(c,fac,rm):
    fitted=corr.fit_drains_to_cornice(c,fac)
    roofs=corr.roof_details(c,rm,fitted,2.4)
    if c["roof"]=="flat":
        out=[coping(c,m) if m["kind"]=="continuous_parapet" else
             access_cap(c,m) if m["kind"]=="access" else m for m in roofs]
        # Check cap against each actual unrelated roof part, not whole hollow ring.
        for m in out:
            if m["kind"]!="access":continue
            for other in out:
                if other["id"]==m["id"]:continue
                for part in other["geometry"]:
                    assert not corr.overlaps(m,dict(geometry=[part])),(m["id"],other["id"])
        return fitted+out
    ext=overhang(c);gut=gutters(c)
    newfac=[bend_drain(c,m) if m["kind"]=="drain" else m for m in fitted]
    # Positive-volume overlap forbidden with unrelated facade modules.
    for m in newfac:
        if m["kind"]=="drain":
            assert all(not corr.overlaps(m,q) for q in newfac if q["id"]!=m["id"])
        for q in (ext,gut):
            for part in q["geometry"]:
                assert not corr.overlaps(m,dict(geometry=[part])),(m["id"],q["id"])
    # The soffit intentionally meets/intersects the original cornice support;
    # outside extensions cannot intersect interior rooftop equipment.
    for m in roofs:
        if m["kind"]=="cornice":continue
        for part in m["geometry"]:
            for q in (ext,gut):
                for pp in q["geometry"]:
                    assert not corr.overlaps(dict(geometry=[part]),dict(geometry=[pp])),(m["id"],q["id"])
    return newfac+roofs+[ext,gut]

def shader(old,cells,styles,joints,enhanced):
    base=corr.shader(old,cells,styles,joints,True,True)
    def shade(s,r,maps,mode,dense=False):
        col,em=base(s,r,maps,mode,dense)
        if enhanced and r.get("accessory")=="access_front":
            u,z=s.T;w,H=r["span"]
            x=np.abs(u-w/2)
            mask=(x<.49)&(z>.045)&(z<min(1.95,H-.12))
            inner=mask&(x<.43)&(z>.09)&(z<min(1.88,H-.19))
            col[mask]=srgb_decode(np.array([.21,.23,.22]))
            col[inner]=srgb_decode(np.array([.41,.43,.40]))
            handle=inner&(np.abs(u-w/2-.32)<.026)&(np.abs(z-1.0)<.07)
            col[handle]=srgb_decode(np.array([.72,.72,.67]))
        return col,em
    return shade

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
        anchors|={o.id for a in p.facades for o in a.openings if o.role=="balcony_door"}
        fitted=corr.fit_drains_to_cornice(c,fac)
        baseline=fitted+corr.roof_details(c,rm,fitted,2.4)
        candidate=improve(c,fac,rm)
        assert {m["id"] for m in baseline}<={m["id"] for m in candidate}
        plans[c["id"]]=(baseline,candidate)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    cameras=[((58,5,21),28.,(.25,-1,.42)),
             ((107,3,16),30.,(.25,-1,.50)),
             ((92.4,0,14.4),95.,(.38,-1,.26)),
             ((107,6,16),24.,(.65,-1,1.8)),
             ((107,6,9),16.,(.3,1,.28))]
    sheet=Image.new("RGB",(1280,640*len(cameras)+30),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLB | stage 13 economy / roof construction details",fill=(0,0,0))
    old=ref.shader;reports=[];before=None
    try:
        for i,name in enumerate(("baseline","construction")):
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
            ref.shader=shader(old,cells,styles,joints,bool(i))
            for r in t0:
                if r["role"]!="wall":continue
                s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
                a,b=shader(old,cells,styles,joints,False)(s,r,maps,"night")
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
                opening_triangles_added=0,abi_samples=samples,cpp_abi=True,
                cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest(),view_delta=delta))
        assert reports[1]["modules"]["pitched_overhang"]==1 and reports[1]["modules"]["open_gutters"]==1
        assert reports[1]["triangles"]>reports[0]["triangles"]
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report=dict(schema="pcg.research.roof-construction.v1",lanes=reports,
        limitations=["rectangular synthetic flat/gable fixtures; one existing south downspout, rear gutter outlet not generated",
        "closed separate slope strips with coincident internal contacts, not a Boolean-unioned roof",
        "gutter-to-pipe contact is visual, no hydraulic flow or cut hole in trough",
        "existing cornice remains a support under new sloped overhang; no full hip/complex-roof extension",
        "access door is shader only; no occupancy addition; no weather/season effects",
        "CPU diffuse diagnostic, no native PBR/hardware performance"])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("ROOF_CONSTRUCTION_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=76,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"ROOF_CONSTRUCTION_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
