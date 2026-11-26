"""Research 19: combine repaired loggias with the actual shared cutout layer.
Reuse tested stage 17 masks/strip/export and stage 18 geometry/PCGF/CPU shadows.
No opening triangles, unique per-building images, or native performance claims.
"""
from pathlib import Path
from collections import Counter,defaultdict
import base64,hashlib,io,json,sys
import numpy as np
from PIL import Image,ImageDraw
from pcg.materials import MATERIALS,texture_maps
from pcg.far_emission import srgb_decode
from render_zero_facade_glsl import raw_mesh
import zero_loggia_repair as repair
import zero_alpha_cutout as cut
import zero_combined_lod as combined
import zero_architecture_correctness as corr
import zero_balcony_guard_variants as variants
import render_zero_geometry_facades as ref

OUT=Path("artifacts/zero-integrated-cutouts")
SOURCE=Path("artifacts/zero-loggia-repair")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def opaque(root):
    v,n,u,c,f=raw_mesh(root/"tile_opaque.glb")
    ids=c[:,0].astype(np.uint32)+256*c[:,1].astype(np.uint32)
    assert np.all(ids[f]==ids[f][:,:1])
    table=json.loads((root/"parameters.json").read_text())
    assert ids.max()<len(table)
    return v[f].reshape(-1,3),u[f].reshape(-1,2),ids[f].reshape(-1),table

def append_layer(scene,v,u,f,kind):
    a,b,c,t=scene
    rid=len(t);name="cutout_"+kind
    r=dict(id=rid,accessory=name,family="painted_rail")
    return (np.concatenate((a,v[f].reshape(-1,3))),
            np.concatenate((b,u[f].reshape(-1,2))),
            np.concatenate((c,np.full(f.size,rid,dtype=c.dtype))),t+[r])

def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    # Stage 17 roof paths already validated against actual stage-16 GLB.
    # Reuse them without recalculating the roof silhouette.
    prior,manifest,*_=cut.source_geometry()
    roof_paths=defaultdict(list)
    for p,r in zip(prior["roof"],[r for r in manifest if r["kind"]=="roof"]):
        bid=r["id"].split(":cutout_guard:")[0]
        roof_paths[bid].append((p,r))
    parts=defaultdict(list);placements=[];removed=Counter();replaced=set()
    def transform(c,mods):
        result=[];offset=np.asarray(c["offset"])
        for m in mods:
            if m["kind"]=="roof_guard":
                paths=roof_paths[c["building_id"]]
                assert paths
                lo,hi=combined.bounds(m)+offset
                new=np.concatenate([p[0] for p,_ in paths])
                assert np.all(new[:,:2].min(0)>=lo[:2]-1e-5)
                assert np.all(new[:,:2].max(0)<=hi[:2]+1e-5)
                assert abs(float(new[:,2].min()-lo[2]))<.03
                for p,r in paths:
                    parts["roof"].append(p)
                    placements.append(dict(r,source_placement=m["id"],building=c["building_id"]))
                removed["roof"]+=m["triangles"];replaced.add(m["id"])
                continue
            # Only ordinary projected open balconies, not the five repaired
            # loggia guards or the opaque full/partial architectural skins.
            rails=[p for p in m["geometry"] if p[0]=="rail"]
            if m["kind"]=="balcony_open":
                assert len(rails)==3
                edges=[];heights=[]
                for _,_,t in rails:
                    vertices=np.unique(t.reshape(-1,3),axis=0)
                    assert len(vertices)==4
                    low,high=vertices[:,2].min(),vertices[:,2].max()
                    bottom=vertices[np.abs(vertices[:,2]-low)<1e-7]
                    assert len(bottom)==2
                    edges.append(tuple(bottom+offset));heights.append(high-low)
                assert np.ptp(heights)<1e-8
                path,closed=cut.chain(edges)
                assert not closed
                p=cut.strip(path,float(heights[0]),cut.PERIOD["balcony"])
                assert len(p[2])==sum(len(t) for _,_,t in rails)
                parts["balcony"].append(p)
                placements.append(dict(id=m["id"]+":cutout",source_placement=m["id"],
                    building=c["building_id"],anchor=m["anchor"],kind="balcony",**p[3]))
                removed["balcony"]+=len(p[2])
                retained=[p for p in m["geometry"] if p[0]!="rail"]
                assert retained
                m=dict(m,geometry=retained,triangles=sum(len(t) for _,_,t in retained))
                m["bounds"]=combined.bounds(m).tolist()
                m["cutout_guard"]=True
            else:
                assert not rails,(m["id"],m["kind"])
            result.append(m)
        return result
    oldout=repair.OUT
    try:
        repair.OUT=OUT
        candidate,rgba,metrics,t0,cells,styles,qs=repair.geometry(decoder,module_transform=transform)
    finally:repair.OUT=oldout
    assert removed["roof"]==912 and len(replaced)==3 and removed["balcony"]>0
    assert set(parts)=={"roof","balcony"}
    assert len({r["id"] for r in placements})==len(placements)
    # The five existing repaired loggia masks remain byte-identical.
    assert sha(OUT/"loggia_cutout.glb")==sha(SOURCE/"loggia_cutout.glb")
    assert (OUT/"facade_cells.bin").read_bytes()==(SOURCE/"facade_cells.bin").read_bytes()
    base=opaque(SOURCE);newopaque=opaque(OUT)
    def by_surface(scene):
        v,u,ids,t=scene;out=defaultdict(list)
        for k,rid in enumerate(ids[::3]):
            r=t[int(rid)]
            out[r["surface"]].append((v[3*k:3*k+3],u[3*k:3*k+3],r))
        return out
    a,b=by_surface(base),by_surface(newopaque)
    assert set(b)<=set(a)
    for k in b:
        assert len(a[k])==len(b[k])
        for (v,u,r),(V,U,R) in zip(a[k],b[k]):
            np.testing.assert_array_equal(v,V);np.testing.assert_array_equal(u,U)
            assert {k:v for k,v in r.items() if k!="id"}=={k:v for k,v in R.items() if k!="id"}
    removed_count=0
    for k in set(a)-set(b):
        for v,u,r in a[k]:
            assert r.get("placement") in replaced or r.get("accessory")=="rail"
            removed_count+=1
    assert removed_count==sum(removed.values())
    assert len(base[0])//3-len(newopaque[0])//3==removed_count
    # Candidate geometry already includes actual reloaded loggia strip
    # triangles and its diagnostic record. Attach identical layer to baseline.
    n=metrics["cutout_triangles"]*3
    assert n==30
    lv,lu=candidate[0][-n:],candidate[1][-n:]
    lf=np.arange(n,dtype=np.uint32).reshape(-1,3)
    baseline=append_layer(base,lv,lu,lf,"rail")
    assert len(baseline[0])//3==3606
    images={k:cut.pattern(k,512) for k in parts}
    meshes={k:cut.join(p) for k,p in parts.items()}
    path=OUT/"roof_balcony_cutout.glb";cut.export(path,meshes,images)
    actual=cut.read(path,meshes,images)
    for k,(v,u,f,im) in actual.items():
        candidate=append_layer(candidate,v,u,f,"roof" if k=="roof" else "projected")
    added={k:len(q[2]) for k,q in actual.items()}
    assert added["roof"]==20 and added["balcony"]==removed["balcony"]
    assert len(candidate[0])//3==3606-912+20
    # Each mask has an independent byte-exact mip chain and native-neutral
    # resource manifest; embedded GLB base image alone cannot store these mips.
    all_images={"rail":rgba,"roof":actual["roof"][3],"projected":actual["balcony"][3]}
    mipstats={}
    for k,im in all_images.items():
        levels,stats=cut.mip_chain(im,True);mipstats[k]=stats
        d=OUT/"mips"/k;d.mkdir(parents=True,exist_ok=True)
        for i,mip in enumerate(levels):Image.fromarray(mip).save(d/f"{i}.png")
    # Real EGL full-scene depth check using actual reloaded GLB and
    # coverage-preserving mips. Different draw order must not alter depth.
    lab=cut.Lab()
    v,u,ids,t=newopaque
    obj=[(lab.mesh(v,np.zeros((len(v),2)),np.arange(len(v),dtype=np.uint32).reshape(-1,3)),None,(.65,.64,.59))]
    masks=[("rail",lv,lu,lf,rgba)]+[
        ("roof" if k=="roof" else "projected",v,u,f,im) for k,(v,u,f,im) in actual.items()]
    for k,v,u,f,im in masks:
        tex=lab.texture(cut.mip_chain(im,True)[0],8.)
        obj.append((lab.mesh(v,u,f),tex,(0,0,0)))
    cam=((61,6,10),4.3,(.35,-1,.45))
    im,d=lab.render(obj,*cam);_,e=lab.render(list(reversed(obj)),*cam)
    np.testing.assert_allclose(d,e,rtol=0,atol=1e-6)
    Image.fromarray(im).save(OUT/"egl_whole_scene.png")
    # CPU facade colors and geometry-derived shadow map from the same
    # reloaded scene. Bilinear masks only here; the EGL lane tests mips.
    joints={r["facade_id"]:corr.panel_joints(r,cells) for r in t0 if r["role"]=="wall" and styles[r["building"]]=="panel_block"}
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    original=ref.shader;base_shade=variants.variant_shader(original,cells,styles,joints)
    alpha={k:repair.alpha_visibility(im) for k,im in all_images.items()}
    def kind(r):return r.get("accessory","").removeprefix("cutout_") if r.get("accessory","").startswith("cutout_") else None
    def visible(s,r):
        k=kind(r)
        return alpha[k](s,dict(r,accessory="cutout_rail")) if k in alpha else np.ones(len(s),bool)
    def two(r):return kind(r) in alpha
    def shade(s,r,maps,mode,dense=False):
        k=kind(r)
        if k not in alpha:return base_shade(s,r,maps,mode,dense)
        rgb=cut.RGB["roof" if k=="roof" else "balcony"]
        col=np.broadcast_to(srgb_decode(np.array(rgb)/255.),(len(s),3)).copy()
        return col,np.zeros_like(col)
    cams=[("whole", (61,6,10),4.3,(.35,-1,.45)),
          ("panel_full",(58,6,10.5),21.,(.35,-1,.22)),
          ("brick_balconies",(107,0,9),32.,(.25,-1,.16)),
          ("roof",(58,5,21),28.,(.25,-1,.42)),
          ("aerial",(61,6,10),3.0,(.3,-1,1.8))]
    sheet=Image.new("RGB",(1280,640*len(cams)),(230,230,230))
    views=[]
    ref.shader=shade
    try:
        for i,(name,center,ppm,view) in enumerate(cams):
            saved=[]
            for j,(lane,scene) in enumerate((("stage18",baseline),("integrated",candidate))):
                lighting,_=repair.shadow_lighting(scene,maps,center,ppm,(.5,-.7,.63),visible,two)
                rid=np.full((640,640),-1,np.int32)
                im,dep=ref.render(*scene,maps,"day",center_override=center,ppm_override=ppm,
                    view_override=view,visibility=visible,two_sided=two,lighting=lighting,capture_ids=rid)
                im.save(OUT/f"{name}_{lane}.png")
                saved.append((np.asarray(im).copy(),dep))
                ImageDraw.Draw(im).text((8,8),lane,fill=(0,0,0));sheet.paste(im,(640*j,640*i))
            (a,da),(b,db)=saved
            views.append(dict(view=name,rgb_changed=int(np.any(a!=b,axis=2).sum()),
                silhouette_xor=int(np.sum(np.isfinite(da)!=np.isfinite(db)))))
        assert sum(q["rgb_changed"] for q in views)>100
        for lane,scene in (("stage18",baseline),("integrated",candidate)):
            im,_=ref.render(*scene,maps,"night",center_override=cam[0],ppm_override=cam[1],
                view_override=cam[2],visibility=visible,two_sided=two)
            im.save(OUT/f"night_{lane}.png")
        for r in t0:
            if r["role"]!="wall":continue
            s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
            for a,b in zip(shade(s,baseline[3][r["id"]],maps,"night"),shade(s,candidate[3][r["id"]],maps,"night")):
                np.testing.assert_array_equal(a,b)
    finally:ref.shader=original
    sheet.save(OUT/"comparison.png")
    (OUT/"cutout_placements.json").write_text(json.dumps(placements,indent=2))
    files=["tile_opaque.glb","loggia_cutout.glb","roof_balcony_cutout.glb","facade_params.bin","facade_cells.bin","parameters.json","placements.json","cutout_placements.json"]
    files+=sorted(str(p.relative_to(OUT)) for p in (OUT/"mips").rglob("*.png"))
    bundle=dict(schema="pcg.research.integrated-cutouts.v1",opaque=files[0],
        masks=files[1:3],hashes={p:sha(OUT/p) for p in files},anchors=metrics["anchors"],
        bounds=[candidate[0].min(0).tolist(),candidate[0].max(0).tolist()])
    (OUT/"integrated_bundle.json").write_text(json.dumps(bundle,indent=2))
    report=dict(schema=bundle["schema"],baseline_triangles=len(baseline[0])//3,
        candidate_triangles=len(candidate[0])//3,removed_opaque=dict(removed),added_cutout=added,
        anchors=metrics["anchors"],recessed_anchors=metrics["recessed_anchors"],opening_triangles_added=0,
        cpp_abi=metrics["cpp_abi"],opaque_records=len(newopaque[3]),views=views,
        egl_draw_order_max_depth_delta=float(np.max(np.abs(d-e))),mips=mipstats,
        limitations=["Three synthetic rectangular buildings only; no GPU or native performance.",
          "Two mask GLBs plus opaque tile, not a final native resource format.",
          "CPU close-up masks bilinear; EGL neutral diagnostic uses coverage mips, not native temporal/shadow acceptance.",
          "Opaque partial/full balcony glazing unchanged; no new window/door geometry.",
          "Rear balcony aperture composition and loggia finishes remain follow-ups; no weather."])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("INTEGRATED_CUTOUT_REPORT "+json.dumps({k:v for k,v in report.items() if k!="mips"}),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=78,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"INTEGRATED_CUTOUT_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
