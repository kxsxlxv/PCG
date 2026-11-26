"""Stage 23: metric architectural trim, drainage and construction-linked aging.
Runs against the exported external-only stage-20 geometry and stage-21A shader.
No changes to canonical wall/opening ABI or pending stage 21/22 work.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, shutil, subprocess
import numpy as np
from PIL import Image, ImageDraw
from pcg.materials import MATERIALS
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table, LAYERS, opening_rectangles
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_combined_lod as combined
import zero_roof_construction as construction
import zero_loggia_repair as repair
import zero_integrated_cutouts as integrated
import zero_window_visual as windows
import zero_facade_trim_geometry as geo

OUT=Path("artifacts/zero-facade-trim")
PREV=Path("artifacts/zero-external-balconies")

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def sample_map(s,maps,family):
    a=maps[family];u,z=s.T;scale=MATERIALS[family].repeat_m
    ix=np.floor((u/scale%1)*a.shape[1]).astype(int)%a.shape[1]
    iy=np.floor((-z/scale%1)*a.shape[0]).astype(int)%a.shape[0]
    return srgb_decode(a[iy,ix].copy())

def noise(u,z,seed,sx,sz):
    # Continuous seeded value noise: no per-pixel independent random grain.
    x=u/sx;y=z/sz;a=np.floor(x);b=np.floor(y)
    t=x-a;q=y-b;t=t*t*(3-2*t);q=q*q*(3-2*q)
    def h(i,j):
        k=np.sin(i*127.1+j*311.7+(int(seed)%65521)*.017)*43758.5453
        return k-np.floor(k)
    return (h(a,b)*(1-t)+h(a+1,b)*t)*(1-q)+(h(a,b+1)*(1-t)+h(a+1,b+1)*t)*q

def material_shader(base,cells,modules,drains,flat=False):
    by=defaultdict(list)
    for m in modules:
        if "span" in m:by[m["facade"]].append(m)
    def shade(s,r,maps,mode,dense=False):
        col,em=base(s,r,maps,mode,dense)
        if not len(s) or mode=="bare":return col,em
        k=r.get("accessory","");u,z=s.T
        if k.startswith("stage23_") or k in ("cornice","bent_drain","gutter_floor","gutter_side","soffit"):
            seed=ref.seed(r.get("placement",r["surface"]))
            n=noise(u,z,seed,.37,.45)
            if k in ("stage23_drain","bent_drain","gutter_floor","gutter_side"):
                # Galvanized/coated metal appearance; not a metallic BRDF.
                col[:]=srgb_decode(np.array([.48,.49,.47]))*(.78+.22*n[:,None])
                if k in ("stage23_drain","bent_drain"):
                    # Bands on the long vertical faces only (UV V is up).
                    if r["span"][1]>1.:
                        band=np.minimum(z%2.1,2.1-z%2.1)<.035
                        col[band]*=.60
            else:
                col*= (.76+.23*n[:,None])
                if r.get("span",[0,0])[1]>.08 and k in ("cornice","stage23_belt","stage23_cornice_under"):
                    f=np.clip(z/max(r["span"][1],.01),0,1)
                    col*= (.82+.18*f[:,None])
            return col,em
        if not windows.is_host_wall(r):return col,em
        role,_,_=tile.lookup(s,r,cells);wall=role==0
        if not wall.any():return col,em
        style=r["building"].split(":")[-1];seed=r["seed"];L=r["length"];H=r["height"];fh=r["floor_h"]
        plinth=geo.PROFILES[style][0]
        n=noise(u,z,seed,1.8,2.4); streak=noise(u,z,seed+11,.18,2.8)
        wet=np.exp(-np.maximum(z,0)/.44)*(.32+.68*noise(u,z,seed+19,.62,.35))
        roof=np.exp(-np.maximum(H-z,0)/.58)*(.30+.70*streak)
        corner=np.exp(-np.maximum(np.minimum(u,L-u),0)/.16)*(.30+.70*n)
        dirt=.065*n+.19*wet+.105*roof+.07*corner
        # The sill above this pixel must be an actual ordinary window.
        if r["bays"]:
            f=np.floor((z-r["sill"])/fh).astype(int)+1
            b=np.floor((u-r["first_center"])/r["spacing"]+.5).astype(int)
            valid=(f>=0)&(f<r["floors"])&(b>=0)&(b<r["bays"])
            ii=np.flatnonzero(valid)
            if len(ii):
                code=cells[r["cell_start"]+f[ii]*r["bays"]+b[ii]]&3
                delta=f[ii]*fh+r["sill"]-z[ii]
                jamb=np.abs(np.abs(u[ii]-(r["first_center"]+b[ii]*r["spacing"]))-r["window_width"]*.43)
                rain=np.exp(-jamb/.11)*np.exp(-np.maximum(delta,0)/.58)
                dirt[ii]+=.15*rain*(.38+.62*streak[ii])*(code==1)
        for x in drains.get(r["id"],[]):
            dirt+=.20*np.exp(-np.abs(u-x)/.18)*(.42+.58*streak)
        mask=wall&(z<plinth+.035)
        if mask.any():
            family="concrete_panel" if style=="panel_block" else "stone_plinth"
            tint={"panel_block":.52,"brick_stack":.71,"mixed_use":.66}[style]
            col[mask]=sample_map(s[mask],maps,family)*tint
            # Damp rises from ground, separate from the repeated stone map.
            col[mask]*=(.78+.22*np.clip(z[mask]/plinth,0,1))[:,None]
        for m in by.get(r["facade_id"],[]):
            a,b=m["span"];lo=min(p[1] for p in m["profile"]);hi=max(p[1] for p in m["profile"])
            span=wall&(u>=a)&(u<=b)
            # Contact/capillary stain below actual mouldings; no false stripe
            # across a skipped window, door or drainage clearance.
            d=lo-z;hit=span&(d>=0)&(d<.30)
            dirt[hit]+=.13*np.exp(-d[hit]/.085)*(.55+.45*streak[hit])
            if flat:
                band=span&(z>=lo)&(z<=hi)
                family=m["geometry"][0][1]
                col[band]=sample_map(s[band],maps,family)*(.72+.23*(z[band]-lo)/(hi-lo))[:,None]
        col[wall]*=(1-np.clip(dirt[wall],0,.46))[:,None]
        return col,em
    return shade

def shadow(scene,maps,center,ppm,visible,two,res=1280):
    direction=(.5,-.7,.63);right,up,view=repair.axes(direction)
    center=np.asarray(center,float);sppm=min(ppm*.55,38.)*res/640.
    _,depth=ref.render(*scene,maps,"day",center_override=center,ppm_override=sppm*640/res,
        view_override=direction,visibility=visible,two_sided=two,resolution=(res,res),depth_only=True)
    def light(world,n,r):
        p=world-center
        x=np.floor(p@right*sppm+res/2).astype(int);y=np.floor(-p@up*sppm+res/2).astype(int)
        zz=p@view;lit=np.zeros(len(world),float)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                X=x+dx;Y=y+dy;valid=(X>=0)&(X<res)&(Y>=0)&(Y<res)
                q=np.ones(len(world),bool);q[valid]=depth[Y[valid],X[valid]]<=zz[valid]+.045
                lit+=q/9.
        return .34+.66*max(0,float(n@view))*lit
    return light

def run(cases,groups,table,cells,scene,actual,maps,base,visible,two,decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    assert all(len(c["outer"])==4 and not c["holes"] for c in cases),"rectangular research scope"
    rect=lambda r:opening_rectangles(r,cells)
    modules,audit,drains,contacts=geo.plan(scene,rect,construction.swept_pipe)
    assert sum(m["kind"]=="stage23_drain" for m in modules)==1
    counts=Counter(m["kind"] for m in modules)
    assert all(counts[k]>0 for k in ("plinth_drip","belt","cornice_under","stage23_drain"))
    # Deterministic plan, same exact triangle arrays on repeated construction.
    again,_,_,_=geo.plan(scene,rect,construction.swept_pipe)
    assert [m["id"] for m in modules]==[m["id"] for m in again]
    for m,n in zip(modules,again):np.testing.assert_array_equal(m["geometry"][0][2],n["geometry"][0][2])
    old_cells=cells.tobytes()
    g={role:{k:v.copy() for k,v in q.items()} for role,q in groups.items()}
    t=list(table);by={c["building_id"]:c for c in cases}
    for m in modules:
        # Geometry planner operates in exported world coordinates.
        c=dict(by[m["building"]],offset=[0,0,0])
        combined.append_compact(g,t,c,m)
    for role,q in groups.items():
        for k,v in q.items():np.testing.assert_array_equal(g[role][k][:len(v)],v)
    assert t[:len(table)]==table and cells.tobytes()==old_cells
    path=OUT/"tile_opaque.glb";tile.export_glb(g,path)
    v,u,ids,_=tile.read_glb(path,g,t)
    samples=pack_table(t,cells,OUT/"facade_params.bin")
    (OUT/"facade_cells.bin").write_bytes(old_cells)
    check=json.loads(subprocess.run([decoder,str(OUT/"facade_params.bin"),str(OUT/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert check["records"]==len(t) and check["cells"]==len(cells)
    candidate=(v,u,ids,t)
    for k,(vv,uu,ff,im) in actual.items():candidate=integrated.append_layer(candidate,vv,uu,ff,k)
    shutil.copyfile(PREV/"roof_balcony_cutout.glb",OUT/"roof_balcony_cutout.glb")
    added=len(candidate[0])//3-len(scene[0])//3
    assert added==sum(len(p) for m in modules for _,_,p in m["geometry"]) and 0<added<=900
    meta=[dict({k:v for k,v in m.items() if k!="geometry"},
        triangles=sum(len(p) for _,_,p in m["geometry"]),bounds=combined.bounds(m).tolist(),
        material=m["geometry"][0][1]) for m in modules]
    dump(OUT/"placements.json",meta);dump(OUT/"parameters.json",t)
    dump(OUT/"clearance.json",audit);dump(OUT/"drains.json",drains)
    baseline=windows.candidate(base,cells)
    near=material_shader(baseline,cells,modules,drains)
    lite=material_shader(baseline,cells,modules,drains,True)
    checked=0;changed=0
    for r in scene[3]:
        if windows.is_host_wall(r):
            s=np.random.default_rng(r["seed"]).random((4096,2))*[r["length"],r["height"]]
            role,_,_=tile.lookup(s,r,cells);opening=role!=0
        elif r.get("accessory","").startswith("cutout_"):
            s=np.array([[.15,.25],[.5,.75],[1.,1.]])
            opening=np.ones(len(s),bool)
        else:continue
        for mode in ("day","night"):
            old=baseline(s,r,maps,mode)
            for shader in (near,lite):
                a=shader(s,r,maps,mode);b=shader(s,r,maps,mode)
                for x,y,z in zip(old,a,b):
                    np.testing.assert_array_equal(x[opening],y[opening])
                    np.testing.assert_array_equal(y,z)
                    assert np.isfinite(y).all() and (y>=0).all()
                changed+=int(np.any(old[0]!=a[0],axis=1).sum())
        checked+=len(s)
    assert checked>0 and changed>0
    cams=[
        ("panel",(58,6,10.6),16.,(.35,-1,.22),"day",(1920,1080)),
        ("brick",(107,6,8.5),17.,(.32,-1,.22),"day",(1920,1080)),
        ("mixed",(15,6,9),16.,(.35,-1,.23),"day",(1920,1080)),
        ("plinth",(54,0,1.8),52.,(.30,-1,.12),"day",(1280,1280)),
        ("rear_drain",(107,12,7.5),18.,(-.25,1,.20),"day",(1920,1080)),
        ("aerial",(61,6,9),4.3,(.3,-1,1.6),"day",(1920,1080)),
        ("night",(58,6,10.6),16.,(.35,-1,.22),"night",(1920,1080)),
    ]
    sheet=Image.new("RGB",(1500,330*len(cams)),(226,228,228));draw=ImageDraw.Draw(sheet)
    original=ref.shader;views=[]
    try:
        for i,(name,center,ppm,view,mode,resolution) in enumerate(cams):
            imgs=[];depths=[]
            lighting0=shadow(scene,maps,center,ppm,visible,two) if mode=="day" else None
            lighting1=shadow(candidate,maps,center,ppm,visible,two) if mode=="day" else None
            for j,(label,sc,shader,light) in enumerate((
                ("baseline",scene,baseline,lighting0),
                ("geometry",candidate,near,lighting1),
                ("shader_only",scene,lite,lighting0))):
                ref.shader=shader
                im,d=ref.render(*sc,maps,mode,center_override=center,ppm_override=ppm,
                    view_override=view,visibility=visible,two_sided=two,lighting=light,resolution=resolution)
                assert im.size==resolution
                im.save(OUT/f"{name}_{label}.png")
                imgs.append(np.asarray(im));depths.append(d)
                thumb=im.copy();thumb.thumbnail((500,300));sheet.paste(thumb,(j*500+(500-thumb.width)//2,i*330+22))
                draw.text((j*500+8,i*330+5),name+" | "+label,fill=(0,0,0))
            np.testing.assert_array_equal(depths[0],depths[2])
            views.append(dict(view=name,resolution=list(resolution),geometry_changed_rgb=int(np.any(imgs[0]!=imgs[1],axis=2).sum()),
                shader_changed_rgb=int(np.any(imgs[0]!=imgs[2],axis=2).sum()),geometry_changed_depth=int(np.count_nonzero(depths[0]!=depths[1]))))
    finally:ref.shader=original
    sheet.save(OUT/"comparison.png")
    assert all(x["geometry_changed_rgb"]>0 and x["shader_changed_rgb"]>0 for x in views)
    report=dict(schema="pcg.research.facade-trim.v1",baseline_triangles=len(scene[0])//3,
        geometry_triangles=len(candidate[0])//3,shader_only_triangles=len(scene[0])//3,
        added_triangles=added,module_counts=dict(counts),per_building_added_triangles={
            k:sum(len(p) for m in modules if m["building"]==k for _,_,p in m["geometry"]) for k in by},
        existing_geometry_prefix_preserved=True,canonical_apertures_and_cells_unchanged=True,
        opening_and_cutout_samples=checked,cpp_abi_samples=samples,miter_overlap_pairs=contacts,
        shadow_map_resolution=1280,views=views,limitations=[
            "Rectangular fixtures only; arbitrary footprint offsets and facade joinery require further tests.",
            "Shader-only comparison is not yet a runtime LOD; new north pipe has no far-lane silhouette.",
            "CPU procedural color/weathering, not a production PBR shader, no native UNIGINE profiling.",
            "Flat-roof drainage is not inferred; existing gable cornice retained to avoid soffit collision.",
            "No new paid textures; full stage 21 and 22 deferred. Actual visual gate pending."])
    dump(OUT/"report.json",report)
    files=[path,OUT/"roof_balcony_cutout.glb",OUT/"facade_params.bin",OUT/"facade_cells.bin",OUT/"placements.json",OUT/"parameters.json",OUT/"clearance.json",OUT/"drains.json"]
    dump(OUT/"bundle.json",dict(schema=report["schema"],hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print("FACADE_TRIM_REPORT "+json.dumps(report),flush=True)
    import zero_roof_detail as roofs
    roofs.run(cases,g,t,cells,candidate,actual,maps,near,visible,two,decoder)
