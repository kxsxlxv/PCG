"""Approved roof revision: shader-only flashing, no round flue sleeves,
ridge trim and batched two-triangle roof/facade antennas.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil,subprocess
import numpy as np
from PIL import Image,ImageDraw
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table,LAYERS
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_facade_trim as trim
import zero_integrated_cutouts as integrated
import zero_alpha_cutout as cut
import zero_loggia_repair as repair
import zero_roof_detail as roofs
import zero_antenna_cards as antennas

OUT=Path("artifacts/zero-roof-refinement")
PREV=Path("artifacts/zero-roof-detail")
REMOVE={"roof24_flue","roof24_flashing"}

def filtered(groups,table):
    dead={r["id"] for r in table if r.get("accessory") in REMOVE}
    out={};removed=Counter()
    for role,g in groups.items():
        ids=ref.decode(g["color"][:,:2].astype(np.float32)/255)
        f=g["f"];rid=ids[f[:,0]]
        keep=~np.isin(rid,list(dead))
        for i in rid[~keep]:removed[table[int(i)]["accessory"]]+=1
        f=f[keep];used=np.unique(f);remap=np.full(len(ids),-1,np.int64);remap[used]=np.arange(len(used))
        out[role]={k:(remap[f].astype(v.dtype) if k=="f" else v[used].copy()) for k,v in g.items()}
        for k,v in g.items():
            if k!="f":np.testing.assert_array_equal(out[role][k],v[used])
    return out,dict(removed)

def revision_shader(base,info,frames):
    original=roofs.surface_shader(base,info,frames,True)
    def shade(s,r,maps,mode,dense=False):
        col,em=original(s,r,maps,mode,dense)
        if not len(s) or mode=="bare":return col,em
        if r["id"] not in frames:return col,em
        D=info[r["building"]]
        if D["flat"]:return col,em
        # Rectangular gable fixture: derive actual ridge height and direction
        # from host coordinates. A metal cap band wraps both slopes, with
        # overlapped seams along the ridge and distinct edge shadow.
        xyz=np.c_[s,np.ones(len(s))]@frames[r["id"]]
        lo,hi=np.asarray(D["bounds"]);x,y=xyz[:,:2].T
        mid=(lo[1]+hi[1])/2
        dist=np.abs(y-mid)
        # Lightweight standing/corrugated metal appearance; actual authored
        # slate maps are evaluated separately when provided locally.
        rib=.5+.5*np.cos(2*np.pi*(x-lo[0])/.18)
        n=trim.noise(x-lo[0],y-lo[1],r["seed"],1.5,2.4)
        sheet=srgb_decode(np.array([.56,.57,.55]))*(.76+.18*rib+.14*n)[:,None]
        # Do not overwrite the actual shader-only flashing.
        flash=np.zeros(len(s),bool)
        for a,b,c,d in D["flash"]:
            q=np.maximum(np.maximum(a-x,x-c),np.maximum(b-y,y-d))
            flash|=(q>=0)&(q<.13)
        col[~flash]=sheet[~flash]
        cap=(dist<.17)&~flash
        col[cap]=srgb_decode(np.array([.65,.67,.66]))*(.82+.18*n[cap,None])
        edge=(dist>=.17)&(dist<.205)&~flash;col[edge]*=.58
        lap=cap&(np.mod(x-lo[0],1.6)<.028);col[lap]*=.65
        return col,em
    return shade

def run(groups,table,cells,scene,actual,maps,base,previous,info,frames,visible,two,decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    g,removed=filtered(groups,table)
    assert removed=={"roof24_flue":384,"roof24_flashing":118},removed
    path=OUT/"tile_opaque.glb";tile.export_glb(g,path)
    v,u,ids,_=tile.read_glb(path,g,table)
    # Exact retained triangle streams, including source UV and semantic ID.
    opaque_count=len(v)//3
    target=(v,u,ids,list(table))
    for k,(vv,uu,ff,im) in actual.items():target=integrated.append_layer(target,vv,uu,ff,k)
    keep=np.array([scene[3][int(i)].get("accessory") not in REMOVE for i in scene[2]])
    for a,b in zip(target[:2],scene[:2]):np.testing.assert_array_equal(a,b[keep])
    old_cells=(PREV/"facade_cells.bin").read_bytes()
    assert old_cells==cells.tobytes()
    (OUT/"facade_cells.bin").write_bytes(old_cells)
    # Keep stable parameter IDs for retained geometry; unused records are
    # intentionally retained so existing PCGF v1 offsets remain identical.
    samples=pack_table(table,cells,OUT/"facade_params.bin")
    assert (OUT/"facade_params.bin").read_bytes()==(PREV/"facade_params.bin").read_bytes()
    check=json.loads(subprocess.run([decoder,str(OUT/"facade_params.bin"),str(OUT/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert check["records"]==len(table) and check["cells"]==len(cells)
    shutil.copyfile(PREV/"roof_balcony_cutout.glb",OUT/"roof_balcony_cutout.glb")
    cards,attempts=antennas.plan(target,cells);again,_=antennas.plan(target,cells)
    assert [m["id"] for m in cards]==[m["id"] for m in again]
    for m,n in zip(cards,again):
        for a,b in zip(m["quad"],n["quad"]):np.testing.assert_array_equal(a,b)
    count=Counter(m["support"] for m in cards);types=Counter(m["type"] for m in cards)
    assert count["roof_plane"]>=6 and count["facade_pier"]>=6 and len(types)>=6,(count,types)
    image=antennas.atlas();Image.fromarray(image).save(OUT/"antenna_atlas.png")
    mesh=antennas.batch(cards);assert len(mesh[2])==2*len(cards)
    p=OUT/"antenna_cutout.glb";cut.export(p,{"antenna":mesh},{"antenna":image})
    vv,uu,ff,rgba=cut.read(p,{"antenna":mesh},{"antenna":image})["antenna"]
    result=integrated.append_layer(target,vv,uu,ff,"antenna")
    alpha=repair.alpha_visibility(rgba)
    def vis(s,r):
        return alpha(s,dict(r,accessory="cutout_rail")) if r.get("accessory")=="cutout_antenna" else visible(s,r)
    def both(r):return r.get("accessory")=="cutout_antenna" or two(r)
    rev=revision_shader(base,info,frames)
    def shade(s,r,maps,mode,dense=False):
        if r.get("accessory")!="cutout_antenna":return rev(s,r,maps,mode,dense)
        W,H=rgba.shape[1],rgba.shape[0]
        x=np.clip(np.floor(s[:,0]*W).astype(int),0,W-1);y=np.clip(np.floor(s[:,1]*H).astype(int),0,H-1)
        c=srgb_decode(rgba[y,x,:3].astype(float)/255.)
        return c,np.zeros_like(c)
    # Protect facade appearance and all alpha-tested guard samples.
    checked=0
    for r in scene[3]:
        if r.get("role")=="wall" and "accessory" not in r:
            s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
        elif r.get("accessory","").startswith("cutout_"):
            s=np.array([[.15,.25],[.5,.75],[1.,1.]])
        else:continue
        for mode in ("day","night"):
            for a,b in zip(previous(s,r,maps,mode),shade(s,r,maps,mode)):
                np.testing.assert_array_equal(a,b)
        checked+=len(s)
    # Render one card from each side with actual EGL alpha discard and
    # double-sided geometry. Transparent pixels must not write depth.
    lab=cut.Lab()
    tex=lab.texture([rgba],1.)
    cardmesh=lab.mesh(vv,uu,ff)
    first=cards[0];q=first["quad"][0];center=q.mean(0)
    d=q[1]-q[0];normal=np.array([d[1],-d[0],0.])
    im0,d0=lab.render([(cardmesh,tex,(0,0,0))],center,100.,normal)
    im1,d1=lab.render([(cardmesh,tex,(0,0,0))],center,100.,-normal)
    assert (d0<1).sum()>10 and (d1<1).sum()>10
    Image.fromarray(im0).save(OUT/"egl_antenna_front.png");Image.fromarray(im1).save(OUT/"egl_antenna_back.png")
    trim.dump(OUT/"placements.json",[{k:v for k,v in m.items() if k!="quad"} for m in cards])
    trim.dump(OUT/"parameters.json",table)
    # Compare identical actual cameras, with fresh shadow depth for each lane.
    dish=next(m for m in cards if m["support"]=="facade_pier")
    roof=next(m for m in cards if m["support"]=="roof_plane")
    cams=[("brick_roof",(107,6,17),23.,(.18,-.7,1.35),(1920,1080)),
          ("panel_roof",(58,6,21.8),25.,(.24,-.65,1.5),(1920,1080)),
          ("aerial",(61,6,10),8.,(.3,-1,1.65),(1920,1080)),
          ("roof_antenna",roof["quad"][0].mean(0),100.,(.22,-1,.22),(1280,1280)),
          ("facade_dish",dish["quad"][0].mean(0),125.,(.30,-1,.10),(1280,1280))]
    # For detail cameras face the actual card rather than viewing edge-on.
    for i,m in ((3,roof),(4,dish)):
        q=m["quad"][0];d=q[1]-q[0];n=np.array([d[1],-d[0],0.]);n/=np.linalg.norm(n)
        if i==4:
            # Quad extends outward; choose facade exterior side based on
            # host normal inferred from its nearest wall plane.
            a=np.asarray(m["anchor"]);walls=[r for r in target[3] if r.get("role")=="wall" and r["building"]==m["building"]]
            import zero_facade_trim_geometry as wg
            frames_wall=[wg.wall_frame(*target[:3],r) for r in walls]
            f=min(frames_wall,key=lambda f:abs((a-f[0])@f[2]))
            if n@f[2]<0:n=-n
        cam=cams[i];cams[i]=(cam[0],cam[1],cam[2],n+[0,0,.16],cam[4])
    sheet=Image.new("RGB",(1500,390*len(cams)),(226,228,228));draw=ImageDraw.Draw(sheet);rows=[]
    old=ref.shader
    try:
        for i,(name,center,ppm,view,res) in enumerate(cams):
            l0=trim.shadow(scene,maps,center,ppm,visible,two)
            l1=trim.shadow(result,maps,center,ppm,vis,both)
            images=[]
            for j,(label,sc,fn,visibility,two_sided,lighting) in enumerate((
                ("baseline",scene,previous,visible,two,l0),("revision",result,shade,vis,both,l1))):
                ref.shader=fn
                im,_=ref.render(*sc,maps,"day",center_override=center,ppm_override=ppm,view_override=view,
                     visibility=visibility,two_sided=two_sided,lighting=lighting,resolution=res)
                im.save(OUT/f"{name}_{label}.png");images.append(np.asarray(im))
                thumb=im.copy();thumb.thumbnail((750,360));sheet.paste(thumb,(j*750+(750-thumb.width)//2,i*390+24))
                draw.text((j*750+8,i*390+5),name+" | "+label,fill=(0,0,0))
            rows.append(dict(view=name,changed_rgb_pixels=int(np.any(images[0]!=images[1],axis=2).sum())))
    finally:ref.shader=old
    sheet.save(OUT/"comparison.png")
    total=len(result[0])//3
    assert total==len(scene[0])//3-sum(removed.values())+2*len(cards)
    report=dict(schema="pcg.research.roof-refinement.v1",baseline_triangles=len(scene[0])//3,
        triangles=total,opaque_triangles=opaque_count,removed_triangles=removed,antenna_count=len(cards),
        antenna_triangles=2*len(cards),per_support=dict(count),per_type=dict(types),placement_attempts=attempts,
        protected_samples=checked,cpp_abi_samples=samples,canonical_cells_identical=True,
        atlas_size=[antennas.SIZE,antennas.SIZE],material_count=1,alpha_test=True,two_sided=True,
        ridge_geometry_triangles=0,views=rows,limitations=[
        "Fixed quad front/back share baked appearance and vanish edge-on; no true parabolic silhouette or depth.",
        "Original fallback atlas, not acquired photo-derived impostors; mip preservation and screen-size LOD deferred.",
        "Metal roof and ridge are CPU shader appearance only, no authored PBR in this CI.",
        "Removed round flue sleeves; original rectangular chimneys retained. Flat drain disks retained.",
        "No final window/loggia or entrance revision; commercial EXRs excluded from repository."])
    trim.dump(OUT/"report.json",report)
    files=[path,p,OUT/"antenna_atlas.png",OUT/"roof_balcony_cutout.glb",OUT/"facade_params.bin",OUT/"facade_cells.bin",OUT/"parameters.json",OUT/"placements.json"]
    trim.dump(OUT/"bundle.json",dict(schema=report["schema"],hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print("ROOF_REFINEMENT_REPORT "+json.dumps(report),flush=True)
