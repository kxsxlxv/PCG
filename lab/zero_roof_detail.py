"""Stage 24: roof details and construction-driven roof finish, paired LOD study."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil,subprocess
import numpy as np
from PIL import Image,ImageDraw
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table,LAYERS
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_combined_lod as combined
import zero_integrated_cutouts as integrated
import zero_facade_trim as trim
import zero_roof_detail_geometry as geo

OUT=Path("artifacts/zero-roof-detail")
PREV=Path("artifacts/zero-facade-trim")

def surface_shader(base,info,frames,shader_only=False):
    def shade(s,r,maps,mode,dense=False):
        col,em=base(s,r,maps,mode,dense)
        if not len(s) or mode=="bare":return col,em
        k=r.get("accessory","");u,z=s.T
        if k.startswith("roof24_"):
            if k=="roof24_drain":
                # Dark grate slots on the actual disk surface, no false
                # alpha hole through the host roof.
                bars=np.mod(u/.065,1)<.27
                col[:]=srgb_decode(np.array([.36,.37,.35]))
                col[bars]=srgb_decode(np.array([.09,.10,.10]))
            elif k=="roof24_flue":
                col[:]=srgb_decode(np.array([.31,.30,.28]))
            else:
                col[:]=srgb_decode(np.array([.49,.51,.50]))
            return col,em
        if r["id"] in frames:
            xyz=np.c_[s,np.ones(len(s))]@frames[r["id"]]
            x,y=xyz[:,:2].T;D=info[r["building"]];lo,hi=np.asarray(D["bounds"])
            x=x-lo[0];y=y-lo[1];seed=ref.seed(r["building"])
            noise=trim.noise(x,y,seed,2.1,1.6)
            if D["flat"]:
                # Membrane rolls run along X; stagger end laps across rows.
                row=np.floor(y/1.05);q=np.mod(y,1.05);v=np.mod(x+(row%2)*2.1,4.2)
                seam=(np.minimum(q,1.05-q)<.022)|(np.minimum(v,4.2-v)<.026)
                col*= (.88+.22*noise[:,None])
                col[seam]*=.68
                for a,b,c,d in D["patches"]:
                    a-=lo[0];c-=lo[0];b-=lo[1];d-=lo[1]
                    hit=(x>=a)&(x<=c)&(y>=b)&(y<=d)
                    edge=hit&(np.minimum(np.minimum(x-a,c-x),np.minimum(y-b,d-y))<.035)
                    col[hit]*=1.27;col[edge]*=.69
                # Perimeter deposits, capillary stains at outlets, no uniform
                # painted black rectangle over the entire roof.
                distance=np.minimum(np.minimum(x,hi[0]-lo[0]-x),np.minimum(y,hi[1]-lo[1]-y))
                col*= (1-.18*np.exp(-np.maximum(distance,0)/.32)*noise)[:,None]
                for X,Y in D["drains"]:
                    d=np.hypot(x-(X-lo[0]),y-(Y-lo[1]))
                    col*= (1-.26*np.exp(-d/.48))[:,None]
                    if shader_only:
                        mask=d<.24;col[mask]=srgb_decode(np.array([.27,.28,.26]))
                # Walkway is a membrane protective layer between access
                # housing and vent bank, not additional full-width geometry.
                path=(np.abs(y-(hi[1]-lo[1])*.5)<.46)&(x>(hi[0]-lo[0])*.31)&(x<(hi[0]-lo[0])*.70)
                col[path]*=1.20
            else:
                # Broad coherent variations in ceramic tile coloration;
                # tile detail remains in existing metric material.
                col*= (.79+.32*noise[:,None])
                ridge=np.abs(y-(hi[1]-lo[1])*.5)<.14
                col[ridge]*=.78
                # Weathering below ridge and chimney stacks.
                col*= (1-.09*np.exp(-np.abs(y-(hi[1]-lo[1])*.5)/.65))[:,None]
            for a,b,c,d in D["flash"]:
                a-=lo[0];c-=lo[0];b-=lo[1];d-=lo[1]
                dist=np.maximum(np.maximum(a-x,x-c),np.maximum(b-y,y-d))
                hit=(dist>=0)&(dist<.13)
                if shader_only:col[hit]=srgb_decode(np.array([.46,.48,.46]))
                col*= (1-.13*np.exp(-np.maximum(dist,0)/.35))[:,None]
            return col,em
        if k=="coping":
            n=trim.noise(u,z,r["seed"],.5,.4)
            col[:]=srgb_decode(np.array([.51,.53,.52]))*(.85+.22*n[:,None])
            col[np.mod(u/1.8,1)<.016]*=.60
        elif k=="vent":
            # Horizontal vent grille on vertical faces; do not recolor the
            # horizontal top or paint holes on the concrete roof itself.
            w,h=r["span"]
            if h>.40 and w>.35:
                grille=(u>w*.18)&(u<w*.82)&(z>h*.55)&(z<h*.83)
                col[grille]=srgb_decode(np.array([.18,.19,.18]))
                col[grille&(np.mod(z/.095,1)>.64)]=srgb_decode(np.array([.42,.43,.42]))
        elif k=="chimney":
            w,h=r["span"];n=trim.noise(u,z,r["seed"],.35,.8)
            col*= (1-.22*np.exp(-np.maximum(h-z,0)/.35)*n)[:,None]
        elif k=="access_cover":
            col*= (.79+.24*trim.noise(u,z,r["seed"],.45,.65)[:,None])
        return col,em
    return shade

def run(cases,groups,table,cells,scene,actual,maps,base,visible,two,decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    modules,info,frames,checks=geo.plan(scene)
    counts=Counter(m["kind"] for m in modules)
    assert counts["flue"]==6 and counts["cowl"]==5 and counts["dormer_sill"]==2
    assert counts["flashing"]==12 and counts["drain"]>0
    again,_,_,_=geo.plan(scene)
    for m,n in zip(modules,again):
        assert m["id"]==n["id"]
        for (_,_,a),(_,_,b) in zip(m["geometry"],n["geometry"]):np.testing.assert_array_equal(a,b)
    g={role:{k:v.copy() for k,v in q.items()} for role,q in groups.items()};t=list(table)
    by={c["building_id"]:c for c in cases}
    for m in modules:combined.append_compact(g,t,dict(by[m["building"]],offset=[0,0,0]),m)
    for role,q in groups.items():
        for k,v in q.items():np.testing.assert_array_equal(g[role][k][:len(v)],v)
    assert t[:len(table)]==table
    path=OUT/"tile_opaque.glb";tile.export_glb(g,path)
    v,u,ids,_=tile.read_glb(path,g,t)
    (OUT/"facade_cells.bin").write_bytes(cells.tobytes())
    assert (OUT/"facade_cells.bin").read_bytes()==(PREV/"facade_cells.bin").read_bytes()
    samples=pack_table(t,cells,OUT/"facade_params.bin")
    check=json.loads(subprocess.run([decoder,str(OUT/"facade_params.bin"),str(OUT/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert check["records"]==len(t) and check["cells"]==len(cells)
    candidate=(v,u,ids,t)
    for k,(vv,uu,ff,im) in actual.items():candidate=integrated.append_layer(candidate,vv,uu,ff,k)
    shutil.copyfile(PREV/"roof_balcony_cutout.glb",OUT/"roof_balcony_cutout.glb")
    added=len(candidate[0])//3-len(scene[0])//3
    assert added==sum(len(p) for m in modules for _,_,p in m["geometry"]) and 0<added<=1000
    meta=[dict({k:v for k,v in m.items() if k!="geometry"},triangles=sum(len(p) for _,_,p in m["geometry"])) for m in modules]
    trim.dump(OUT/"placements.json",meta);trim.dump(OUT/"parameters.json",t);trim.dump(OUT/"roof_layout.json",info)
    near=surface_shader(base,info,frames);lite=surface_shader(base,info,frames,True)
    protected=0;roof_samples=0
    for r in scene[3]:
        if r.get("role")=="wall" and "accessory" not in r:
            s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
        elif r.get("accessory","").startswith("cutout_"):s=np.array([[.15,.25],[.5,.75],[1.,1.]])
        elif r["id"] in frames:
            s=scene[1][scene[2]==r["id"]].astype(float)
            roof_samples+=len(s)
        else:continue
        for mode in ("day","night"):
            a=base(s,r,maps,mode)
            for fn in (near,lite):
                b=fn(s,r,maps,mode);c=fn(s,r,maps,mode)
                for x,y,z in zip(a,b,c):
                    np.testing.assert_array_equal(y,z)
                    assert np.isfinite(y).all() and (y>=0).all()
                    if r["id"] not in frames:np.testing.assert_array_equal(x,y)
        if r["id"] not in frames:protected+=len(s)
    cams=[
        ("panel_roof",(58,6,21.8),25.,(.24,-.65,1.5),(1920,1080)),
        ("mixed_roof",(15,6,18.9),23.,(-.24,-.65,1.5),(1920,1080)),
        ("brick_roof",(107,6,17),23.,(.18,-.7,1.35),(1920,1080)),
        ("chimneys",(107,6,18),60.,(.30,-.85,.9),(1280,1280)),
        ("vent_detail",(62.6,6,22),66.,(.25,-.6,1.2),(1280,1280)),
        ("aerial",(61,6,10),8.,(.3,-1,1.65),(1920,1080)),
    ]
    sheet=Image.new("RGB",(1500,330*len(cams)),(226,228,228));draw=ImageDraw.Draw(sheet);rows=[]
    original=ref.shader
    try:
        for i,(name,center,ppm,view,res) in enumerate(cams):
            l0=trim.shadow(scene,maps,center,ppm,visible,two);l1=trim.shadow(candidate,maps,center,ppm,visible,two)
            imgs=[];depths=[]
            for j,(label,sc,fn,light) in enumerate((("baseline",scene,base,l0),("geometry",candidate,near,l1),("shader_only",scene,lite,l0))):
                ref.shader=fn
                im,d=ref.render(*sc,maps,"day",center_override=center,ppm_override=ppm,view_override=view,
                    visibility=visible,two_sided=two,lighting=light,resolution=res)
                assert im.size==res;im.save(OUT/f"{name}_{label}.png")
                imgs.append(np.asarray(im));depths.append(d)
                thumb=im.copy();thumb.thumbnail((500,300));sheet.paste(thumb,(j*500+(500-thumb.width)//2,i*330+22))
                draw.text((j*500+8,i*330+5),name+" | "+label,fill=(0,0,0))
            np.testing.assert_array_equal(depths[0],depths[2])
            rows.append(dict(view=name,resolution=list(res),geometry_changed_rgb=int(np.any(imgs[0]!=imgs[1],axis=2).sum()),
                shader_changed_rgb=int(np.any(imgs[0]!=imgs[2],axis=2).sum()),geometry_changed_depth=int(np.count_nonzero(depths[0]!=depths[1])),
                geometry_vs_shader_mean_abs_rgb=float(np.abs(imgs[1].astype(float)-imgs[2]).mean())))
    finally:ref.shader=original
    sheet.save(OUT/"comparison.png")
    assert all(r["geometry_changed_rgb"]>0 and r["shader_changed_rgb"]>0 for r in rows)
    report=dict(schema="pcg.research.roof-detail.v1",baseline_triangles=len(scene[0])//3,
        geometry_triangles=len(candidate[0])//3,shader_only_triangles=len(scene[0])//3,added_triangles=added,
        module_counts=dict(counts),per_building_added_triangles={b:sum(len(p) for m in modules if m["building"]==b for _,_,p in m["geometry"]) for b in by},
        clearance_checks=checks,protected_samples=protected,roof_vertex_samples=roof_samples,cpp_abi_samples=samples,
        existing_geometry_prefix_preserved=True,cells_unchanged=True,views=rows,limitations=[
        "Three rectangular fixtures; roof detailing on actual planes, not arbitrary GIS roof typology.",
        "Flat roof outlets are visible proxies; no hydraulic calculation, roof falls or internal plumbing.",
        "Existing roof access bodies, parapets, cutout guards and dormant window grammar retained.",
        "CPU material approximation, not PBR/UNIGINE runtime. Shader-only lane lacks new silhouette geometry.",
        "Visual acceptance required on actual CI images. Stage 21/22 remain deferred."])
    trim.dump(OUT/"report.json",report)
    files=[path,OUT/"roof_balcony_cutout.glb",OUT/"facade_cells.bin",OUT/"facade_params.bin",OUT/"placements.json",OUT/"parameters.json",OUT/"roof_layout.json"]
    trim.dump(OUT/"bundle.json",dict(schema=report["schema"],hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print("ROOF_DETAIL_REPORT "+json.dumps(report),flush=True)
    import zero_roof_refinement as refine
    refine.run(g,t,cells,candidate,actual,maps,base,near,info,frames,visible,two,decoder)
