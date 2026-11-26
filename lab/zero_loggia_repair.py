"""Research 18: repair rejected stage-16 loggia visuals.
Reuse stage-16 recess topology/ABI and stage-17 cutout GLB/texture.
CPU diagnostic with actual shadow depth map; not PBR or native acceptance.
"""
from pathlib import Path
import base64,copy,hashlib,io,json,sys,subprocess
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import box
from pcg.architecture import BALCONY_SLAB
from pcg.materials import MATERIALS,texture_maps
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table,LAYERS,opening_rectangles
from render_zero_facade_glsl import raw_mesh
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_facade_architecture as arch
import zero_roof_budgets as roof
import zero_combined_lod as combined
import zero_architecture_correctness as corr
import zero_roof_construction as construction
import zero_balcony_guard_variants as variants
import zero_recessed_loggias as oldlog
import zero_alpha_cutout as cut

OUT=Path("artifacts/zero-loggia-repair")
SOURCE=Path("artifacts/zero-recessed-loggias/depth_1.0")
DEPTH=1.25
RAIL_INSET=.16
GLAZE_INSET=.42

def widen(q,cells):
    r=q["r"];x,z,X,Z=q["cut"].bounds
    # Snap the near-full-bay jambs inward to the 1/8 m metric grid.
    # Fixture has 2.8m bay, 1.4m canonical apertures. Keep the rear
    # aperture geometry/identity exactly as before.
    center=r["first_center"]
    left=np.ceil((center-r["spacing"]/2)*8)/8
    right=np.floor((center+r["spacing"]/2)*8)/8
    top=min(r["height"]-.25,Z+.25)
    b=box(left,z,right,top)
    assert b.area>q["cut"].area and b.covers(q["cut"])
    assert right-left>=.90*r["spacing"]
    assert box(0,0,r["length"],r["height"]).buffer(-oldlog.CLEAR).covers(b)
    want={k for k,_ in q["slots"]}
    for j,(_,bounds) in opening_rectangles(r,cells).items():
        a=box(*bounds)
        if j in want:assert b.buffer(-oldlog.CLEAR+1e-8).covers(a)
        else:assert b.intersection(a.buffer(oldlog.CLEAR-1e-8,join_style=2)).area<1e-9
    return dict(q,cut=b)

def new_recess(groups,table,c,q):
    reveal,mods=oldlog.recess(groups,table,c,q,DEPTH)
    pos,t,n,up=oldlog.frame(q["a"]);x,z,X,Z=q["cut"].bounds
    result=[];strips=[];records=[]
    # Distinct forward architectural envelope vs rear wall opening.
    # Glass remains opaque material illusion per research constraints,
    # but inset from the front so jambs and slab soffit are visible.
    def face(lo,hi,inset,a=x,b=X):
        return oldlog.quad(pos(np.array([[a,lo],[b,lo],[b,hi],[a,hi]]),inset),n)
    for m in mods:
        base=combined.bounds(m)[0,2];sill=base+BALCONY_SLAB
        family=m["envelope_variant"]
        parts=[p for p in m["geometry"] if p[0]=="slab"]
        assert parts and family in ("open","full","partial")
        top=min(base+2.72,Z-.10)
        assert top>sill+.95
        if family=="open":
            a=pos(np.array([[x,sill],[X,sill]]),RAIL_INSET)+np.asarray(c["offset"])
            qstrip=cut.strip(a,.90,cut.PERIOD["balcony"])
            strips.append(qstrip)
            records.append(dict(id=m["id"]+":guard",anchor=m["anchor"],inset_m=RAIL_INSET,
                height_m=.90,triangles=2,**qstrip[3]))
        elif family=="full":
            parts.append(("glazed","painted_rail",face(sill,top,GLAZE_INSET)))
        else:
            mid=sill+.90
            parts.extend([("partial_lower","painted_rail",face(sill,mid,GLAZE_INSET)),
                          ("partial_upper","painted_rail",face(mid,top,GLAZE_INSET,x,(x+X)/2))])
        new=dict(m,geometry=parts,triangles=sum(len(t0) for _,_,t0 in parts),
            enclosure_inset_m=RAIL_INSET if family=="open" else GLAZE_INSET,
            niche_width_m=X-x)
        new["bounds"]=combined.bounds(new).tolist()
        vv=np.concatenate([t0.reshape(-1,3) for _,_,t0 in parts])
        local=(vv-np.r_[q["a"].start,0.])@n
        assert local.max()<=1e-8 and local.min()>=-DEPTH-1e-8
        result.append(new)
    return reveal,result,strips,records

def geometry(decoder):
    cases=arch.fixtures();styles={c["building_id"]:c["id"] for c in cases}
    op=tile.plan_facades
    try:
        tile.plan_facades=arch.styled_plan
        groups,t0,cells,owners=tile.compile_scene(cases,audit_samples=2048)
    finally:tile.plan_facades=op
    table=copy.deepcopy(t0);placements=[];anchors=set();qs_all=[];strips=[];guards=[]
    assert cells.tobytes()==(SOURCE/"facade_cells.bin").read_bytes()
    for c in cases:
        p,fac,fail=arch.plan_modules(c);rm,rf=roof.plan(c)
        assert not fail and not rf
        base=variants.variants(c,p,construction.improve(c,fac,rm),True)
        anchors|={m["anchor"] for m in base if m["kind"].startswith("balcony_")}
        qs=[widen(q,cells) for q in oldlog.plan_recesses(c,p,base,t0,cells)]
        mods=base
        if qs:
            selected=set.union(*(q["selected"] for q in qs))
            mods=[m for m in base if m["id"] not in selected]
            cavity=[]
            for q in qs:
                x,z,X,Z=q["cut"].bounds;pos,_,_,_=oldlog.frame(q["a"])
                uv=np.array([[x,z],[X,z],[X,Z],[x,Z]])
                vv=np.concatenate([pos(uv),pos(uv,DEPTH)])
                lo,hi=vv.min(0),vv.max(0)
                for l,h in cavity:assert not np.all(np.minimum(hi,h)-np.maximum(lo,l)>1e-8)
                cavity.append((lo,hi))
                for m in mods:
                    for _,_,t in m["geometry"]:
                        v=t.reshape(-1,3);l=v.min(0);h=v.max(0)
                        assert not np.all(np.minimum(hi,h)-np.maximum(lo,l)>1e-8),(m["id"],"expanded cavity clash")
            for q in qs:
                reveal,loggias,rail,records=new_recess(groups,table,c,q)
                mods += [reveal]+loggias;strips.extend(rail);guards.extend(records)
                qs_all.append((c,q))
        for m in mods:
            combined.append_compact(groups,table,c,m)
            placements.append({k:v for k,v in m.items() if k!="geometry"})
    assert len(anchors)==86 and len(qs_all)==2 and len(guards)==5
    assert {m["anchor"] for m in placements if m["kind"].startswith("balcony_") or m["kind"]=="recessed_loggia"}==anchors
    assert table[:len(t0)]==t0
    assert len({m["id"] for m in placements})==len(placements)
    path=OUT/"tile_opaque.glb";tile.export_glb(groups,path);v,uv,ids,_=tile.read_glb(path,groups,table)
    host,ht=oldlog.host_mesh(v,ids,table,"facade:panel_block")
    removed=sum(q["cut"].area*DEPTH for _,q in qs_all)
    assert abs(6048.-host.volume-removed)<1e-4
    for c,q in qs_all:oldlog.audit_projection(v,uv,ids,q,cells,DEPTH,np.asarray(c["offset"]))
    (OUT/"facade_cells.bin").write_bytes(cells.tobytes())
    samples=pack_table(table,cells,OUT/"facade_params.bin")
    check=json.loads(subprocess.run([decoder,str(OUT/"facade_params.bin"),str(OUT/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert check["records"]==len(table) and check["cells"]==len(cells)
    (OUT/"parameters.json").write_text(json.dumps(table,indent=2))
    (OUT/"placements.json").write_text(json.dumps(placements,indent=2))
    # Export an actual separate masked GLB, using stage-17 material/UV
    # roundtrip. No placeholder opaque rail in the canonical tile.
    image=cut.pattern("balcony",512);meshes={"balcony":cut.join(strips)}
    cutpath=OUT/"loggia_cutout.glb"
    cut.export(cutpath,meshes,{"balcony":image})
    cv,cu,cf,rgba=cut.read(cutpath,meshes,{"balcony":image})["balcony"]
    assert len(cf)==2*len(guards)
    # Independent actual EGL front/back/depth proof on the same pattern.
    # Existing stage-17 depth_test expects a roof period; use it for the
    # same generic discard semantics, with the balcony texture.
    lab=cut.Lab();tex=lab.texture(cut.mip_chain(rgba,True)[0])
    a,u,f,_=cut.strip([(0,0,0),(2.4,0,0)],.90,cut.PERIOD["balcony"]);u[:,1]=1-u[:,1]
    front=lab.mesh(a,u,f);back=lab.mesh(a+[0,.3,0],u,f)
    cam=((1.2,0,.45),200.,(0,-1,0))
    _,d0=lab.render([(back,None,(.8,.2,.1))],*cam)
    _,df=lab.render([(front,tex,(0,0,0))],*cam)
    im,d=lab.render([(front,tex,(0,0,0)),(back,None,(.8,.2,.1))],*cam)
    bars=df<1;holes=(d0<1)&~bars
    assert bars.sum()>100 and holes.sum()>100 and not np.any(bars&~(d0<1))
    np.testing.assert_array_equal(d[bars],df[bars])
    np.testing.assert_array_equal(d[holes],d0[holes])
    assert np.all(d[bars]<d0[bars])
    Image.fromarray(im).save(OUT/"discard_depth.png")
    depth_proof=dict(front_pixels=int(bars.sum()),through_hole_pixels=int(holes.sum()))
    # The diagnostic-only appended record is NOT serialized into PCGF.
    guard_id=len(table);vis_table=table+[dict(id=guard_id,accessory="cutout_rail",family="painted_rail")]
    scene=(np.concatenate((v,cv[cf].reshape(-1,3))),
           np.concatenate((uv,cu[cf].reshape(-1,2))),
           np.concatenate((ids,np.full(cf.size,guard_id,dtype=ids.dtype))),vis_table)
    assert len(scene[0])//3==len(v)//3+len(cf)
    manifest=dict(schema="pcg.research.loggia-cutout-layer.v1",opaque=path.name,cutout=cutpath.name,
        guards=guards,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,cutpath,OUT/"facade_cells.bin",OUT/"facade_params.bin")})
    (OUT/"bundle.json").write_text(json.dumps(manifest,indent=2))
    metrics=dict(opaque_triangles=len(v)//3,cutout_triangles=len(cf),triangles=len(scene[0])//3,
        host_triangles=ht,host_volume_m3=float(host.volume),removed_volume_m3=removed,
        niche_widths_m=[q["cut"].bounds[2]-q["cut"].bounds[0] for _,q in qs_all],
        anchors=len(anchors),recessed_anchors=12,open_cutout_guards=len(guards),abi_samples=samples,cpp_abi=True,
        opening_triangles_added=0,depth_proof=depth_proof)
    return scene,rgba,metrics,t0,cells,styles,qs_all

def baseline():
    v,n,u,c,f=raw_mesh(SOURCE/"tile.glb")
    ids=c[:,0].astype(np.uint32)+256*c[:,1].astype(np.uint32)
    return v[f].reshape(-1,3),u[f].reshape(-1,2),ids[f].reshape(-1),json.loads((SOURCE/"parameters.json").read_text())

def axes(view):
    v=np.asarray(view,float);v/=np.linalg.norm(v)
    r=np.cross([0,0,1],v);r/=np.linalg.norm(r)
    return r,np.cross(v,r),v

def alpha_visibility(rgba):
    # Bilinear sample of actual reloaded shared base mask, glTF UV
    # origin and wrap conventions. Close-up diagnostic, not mip LOD.
    H,W=rgba.shape[:2];alpha=rgba[:,:,3].astype(float)/255.
    def visible(s,r):
        if r.get("accessory")!="cutout_rail":return np.ones(len(s),bool)
        x=s[:,0]*W-.5;y=np.clip(s[:,1]*H-.5,0,H-1)
        ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
        a=x-ix;b=y-iy;X=(ix+1)%W;Y=np.minimum(iy+1,H-1);ix%=W
        v=(1-a)*(1-b)*alpha[iy,ix]+a*(1-b)*alpha[iy,X]+(1-a)*b*alpha[Y,ix]+a*b*alpha[Y,X]
        return v>=.5
    return visible

def shadow_lighting(scene,maps,center,ppm,light,visibility,two):
    # Orthographic depth map using the SAME actual mesh rasterizer and
    # alpha discard. No baked shadow texture or arbitrary painted bands.
    right,up,view=axes(light);center=np.asarray(center,float)
    # Wider map covers nearby occluders beyond the camera crop.
    sppm=min(ppm*.55,38.)
    _,depth=ref.render(*scene,maps,"day",center_override=center,ppm_override=sppm,
                      view_override=light,visibility=visibility,two_sided=two)
    def shade(world,n,r):
        p=world-center
        xx=(p@right)*sppm+320;yy=-(p@up)*sppm+320
        zz=p@view
        x=np.floor(xx).astype(int);y=np.floor(yy).astype(int)
        lit=np.zeros(len(world),float)
        # 3x3 PCF and metric bias to reduce self-shadowing on split faces.
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                X=x+dx;Y=y+dy;valid=(X>=0)&(X<640)&(Y>=0)&(Y<640)
                q=np.ones(len(world),bool)
                q[valid]=depth[Y[valid],X[valid]]<=zz[valid]+.065
                lit+=q/9.
        return .34+.66*max(0,float(n@view))*lit
    return shade,depth

def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    new,rgba,metrics,t0,cells,styles,qs=geometry(decoder)
    old=baseline()
    assert len(old[0])//3==3606
    joints={r["facade_id"]:corr.panel_joints(r,cells) for r in t0 if r["role"]=="wall" and styles[r["building"]]=="panel_block"}
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    original=ref.shader;base=variants.variant_shader(original,cells,styles,joints)
    def shader(s,r,maps,mode,dense=False):
        if r.get("accessory")=="cutout_rail":
            color=np.broadcast_to(srgb_decode(np.array(cut.RGB["balcony"])/255.),(len(s),3)).copy()
            return color,np.zeros_like(color)
        return base(s,r,maps,mode,dense)
    def two(r):return r.get("accessory")=="cutout_rail"
    visible=alpha_visibility(rgba)
    # Four tight views target actual cut stack center, not generic
    # building auto-fit or a neighboring balcony column.
    cams=[("front_oblique",(48.2,.5,10.5),78.,(.75,-1,.16),(.5,-.7,.63)),
          ("front_straight",(48.2,.5,10.5),95.,(0,-1,.04),(.5,-.7,.63)),
          ("front_below",(48.2,.5,8.5),85.,(.4,-1,-.28),(.5,-.7,.63)),
          ("rear_oblique",(67.8,11.5,11.5),78.,(-.75,1,.26),(-.5,.7,.63)),
          ("front_above",(48.2,.5,15.),80.,(.5,-1,.55),(.5,-.7,.63))]
    lanes=("old_diffuse","old_shadow","new_diffuse","new_shadow")
    sheet=Image.new("RGB",(640*len(lanes),640*len(cams)+28),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLB | same cameras | old diffuse / old shadow / wider inset cutout / wider inset cutout + shadow",fill=(0,0,0))
    ref.shader=shader;report=[]
    try:
        for i,(name,center,ppm,view,light) in enumerate(cams):
            records=[]
            for j,lane in enumerate(lanes):
                scene=old if j<2 else new;vis=None if j<2 else visible
                lighting=None
                if j%2:
                    lighting,light_depth=shadow_lighting(scene,maps,center,ppm,light,vis,two)
                rid=np.full((640,640),-1,np.int32)
                im,dep=ref.render(*scene,maps,"day",center_override=center,ppm_override=ppm,view_override=view,
                     visibility=vis,lighting=lighting,capture_ids=rid,two_sided=two)
                im.save(OUT/f"{name}_{lane}.png")
                ImageDraw.Draw(im).text((8,8),lane,fill=(0,0,0));sheet.paste(im,(j*640,i*640+28))
                r,t,vv=axes(view)
                yy,xx=np.mgrid[:640,:640]
                world=np.asarray(center)+(xx[...,None]+.5-320)/ppm*r-(yy[...,None]+.5-320)/ppm*t+np.where(np.isfinite(dep),dep,0)[...,None]*vv
                rear=np.zeros_like(rid,bool);reveal=np.zeros_like(rid,bool)
                for c,q in qs:
                    pos,_,n,_=oldlog.frame(q["a"])
                    d=(world-np.r_[q["a"].start,0.]-np.asarray(c["offset"]))@n
                    rear|=(rid==q["r"]["id"]) & (np.abs(d+(1. if j<2 else DEPTH))<.001)
                    ids=[r0["id"] for r0 in scene[3] if r0.get("placement")==q["a"].id+":recess"]
                    reveal|=np.isin(rid,ids)
                records.append(dict(lane=lane,rear_wall_pixels=int(rear.sum()),reveal_pixels=int(reveal.sum())))
                if j==3:
                    # Independent solid-vs-cutout test on actual camera.
                    # The exact same GLB triangles are rendered twice;
                    # count rear wall and reveal pixels exposed by holes.
                    solid=np.full((640,640),-1,np.int32)
                    _,sd=ref.render(*scene,maps,"day",center_override=center,ppm_override=ppm,
                         view_override=view,capture_ids=solid,two_sided=two)
                    guard=len(scene[3])-1
                    through=(solid==guard)&(rid!=guard)&(rid>=0)
                    assert np.all(dep[through]<=sd[through]+1e-7)
                    records[-1].update(guard_visible_pixels=int((rid==guard).sum()),
                        through_hole_pixels=int(through.sum()),
                        revealed_rear_or_jamb_pixels=int((through&(rear|reveal)).sum()))
                    diagnostic=np.zeros((640,640,3),np.uint8)
                    diagnostic[rid>=0]=[95,95,95];diagnostic[reveal]=[232,161,67]
                    diagnostic[rear]=[60,158,230];diagnostic[rid==guard]=[55,205,100]
                    Image.fromarray(diagnostic).save(OUT/f"{name}_semantic_depth.png")
            report.append(dict(view=name,lanes=records))
        assert sum(q["lanes"][-1]["revealed_rear_or_jamb_pixels"] for q in report)>100
        assert sum(q["lanes"][-1]["reveal_pixels"] for q in report)>100
        # Compare actual night emission function before/after, independent
        # of visible occlusion by opaque forward glazing.
        for r in t0:
            if r["role"]!="wall":continue
            s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
            a,b=base(s,r,maps,"night");x,y=shader(s,new[3][r["id"]],maps,"night")
            np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,y)
    finally:ref.shader=original
    sheet.save(OUT/"contact.png")
    result=dict(schema="pcg.research.loggia-repair.v1",baseline_triangles=len(old[0])//3,
        geometry=metrics,views=report,limitations=[
        "Same two synthetic rectangular panel stacks only; actual bay widths for this fixture, not arbitrary geometry.",
        "Actual shadow depth maps in CPU diffuse diagnostic; no physical ambient occlusion, PBR, transparency or native acceptance.",
        "Full/partial front enclosure uses opaque material illusion, recessed from facade; it can still occlude rear emission.",
        "Actual alpha MASK GLB separate layer for five open loggias; CPU close-up bilinear alpha and separate EGL depth proof, not integrated native shader.",
        "Existing roof guards and other balcony envelopes unchanged in this experiment; stage-17 full scene cutout integration separate.",
        "Semantic depth colors identify visible rear wall/jamb/guard, not photographic evidence. No weather or snow."])
    (OUT/"report.json").write_text(json.dumps(result,indent=2))
    print("LOGGIA_REPAIR_REPORT "+json.dumps(result),flush=True)
    # Compact close-up comparison, prioritize readable two-column preview.
    thumb=Image.new("RGB",(1280,640*len(cams)))
    for i,(name,*_) in enumerate(cams):
        for j,lane in enumerate(("old_shadow","new_shadow")):
            im=Image.open(OUT/f"{name}_{lane}.png")
            ImageDraw.Draw(im).text((8,8),lane,fill=(0,0,0));thumb.paste(im,(640*j,640*i))
    thumb.save(OUT/"shadow_comparison.png")
    thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=78,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"LOGGIA_REPAIR_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
