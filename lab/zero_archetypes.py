"""Stage 25: actual packed shell + complete external balcony grammar, SDK-free.
The preceding GPU material experiment remains unchanged. This lane tests
architectural composition with the shared PCGF v1 CPU reference, not PBR.
"""
from pathlib import Path
from dataclasses import asdict
from collections import Counter, defaultdict
import hashlib
import json
import math
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint
from pcg.footprint import Footprint
from pcg.archetypes import ARCHETYPES, REGIONS, plan_variant
from pcg.architecture import BALCONY_DEPTH, BALCONY_BASE, BALCONY_SLAB, BALCONY_HEIGHT
from pcg.assets import plan_assets, asset_triangles
from pcg.materials import MATERIALS, texture_maps
from pcg.occupancy import EMISSION
from pcg.far_emission import srgb_decode
import compile_zero_geometry_tile as tile
from compile_zero_tiles import pack_table, LAYERS, new_float32_audit
import render_zero_geometry_facades as ref
from zero_facade_architecture import panel_quad
from zero_combined_lod import append_compact

OUT = Path("artifacts/zero-archetypes")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def rect(w,d):
    return [(0,0),(w,0),(w,d),(0,d)]


def fixtures():
    cases=[]
    def add(key, use, floors, top, outer, holes=(), region="generic", archetype=None, **source):
        f=Footprint("stage25:"+key,outer,holes)
        v=plan_variant(f,use,floors,top,region=region,archetype=archetype,**source)
        i=len(cases)
        cases.append(dict(id=key,building_id=f.building_id,outer=f.exterior,holes=f.holes,
            profile=use,floors=floors,eave_m=v.eave_m,roof=v.mass.actual_roof,rise=top-v.eave_m,
            offset=[(i%6)*60.,(i//6)*52.,0.],variant=v,source=source))
    for i,a in enumerate(ARCHETYPES.values()):
        n,top,w,d={"house":(2,9.5,14,10),"apartments":(5,18.5,30,14),"office":(4,14.8,24,16)}[a.use]
        add(a.id+"_a",a.use,n,top,rect(w,d),region=a.regions[0],archetype=a.id)
    for i,a in enumerate(ARCHETYPES.values()):
        n,w,d={"house":(1+i%3,12.+i%3*2.,9.+i%2*2.),
               "apartments":(5+i%4,32.+i%3*4.,14.+i%2*3.),
               "office":(5+i%3,24.+i%3*4.,15.+i%2*3.)}[a.use]
        n=max(a.levels[0],min(n,a.levels[1]));top=n*3.35+(2.6 if a.roof!="flat" else 0.)
        outer=rect(w,d);holes=[];source={}
        if a.id=="baltic_timber":outer=[(0,0),(w,0),(w,d/2),(w/2,d/2),(w/2,d),(0,d)]
        elif a.id=="perimeter_block":holes=[[(w*.27,d*.25),(w*.73,d*.25),(w*.73,d*.75),(w*.27,d*.75)]];source["party_edges"]=((0,1),)
        elif a.id=="caucasus_house":source.update(roof="hip",eave_m=top-2.0)
        elif a.id=="central_townhouse":source.update(roof="flat",eave_m=top)
        if i%3==1 and not holes:
            angle=math.radians(17);c,s=math.cos(angle),math.sin(angle)
            outer=[(x*c-y*s,x*s+y*c) for x,y in outer]
        add(a.id+"_b",a.use,n,top,outer,holes,region=a.regions[0],archetype=a.id,**source)
    for i,region in enumerate(REGIONS):
        use=("house","apartments","office")[i%3]
        n,top,w,d={"house":(2,9.5,14,10),"apartments":(5,18.5,30,14),"office":(4,14.8,24,16)}[use]
        add("auto_"+region,use,n,top,rect(w,d),region=region)
    return cases


def modules(c):
    f=Footprint(c["building_id"],c["outer"],c["holes"]);v=c["variant"];p=v.plan
    assets=plan_assets(f,p);recipes={r.id:r for r in assets.recipes}
    openings={o.id:(wall,o) for wall in p.facades for o in wall.openings if o.role=="balcony_door"}
    out=[];occupied=[];checks=0
    for q in assets.placements:
        recipe=recipes[q.asset_id]
        if recipe.family!="balcony":continue
        wall,o=openings[q.opening_id];w=recipe.width;d=BALCONY_DEPTH
        glazed=v.glazed(wall.id,o.id)
        floor=p.floor_levels[o.floor];ceil=p.floor_levels[o.floor+1]
        high=min(ceil-floor-BALCONY_BASE-.025,max(2.4,o.z+o.height-floor-BALCONY_BASE+.025)) if glazed else BALCONY_HEIGHT
        if glazed:assert floor+BALCONY_BASE+high>o.z+o.height
        slab=np.array([t for role,t in asset_triangles(recipe,1) if role=="balcony_slab"])
        assert len(slab)==12
        a,b,cc,e=(-w/2,0.),(-w/2,-d),(w/2,-d),(w/2,0.)
        parts=[("slab","concrete_panel",slab)]
        for x,y in ((a,b),(b,cc),(cc,e)):
            parts.append(("glazed" if glazed else "rail","painted_rail",panel_quad(x,y,BALCONY_SLAB,high)))
        if glazed:
            cap=np.array([[*b,high],[*cc,high],[*e,high],[*a,high]])
            parts.append(("slab","concrete_panel",cap[[[0,1,2],[0,2,3]]]))
        geo=[(k,mat,np.array([[q.point(x) for x in t] for t in tris])) for k,mat,tris in parts]
        xyz=np.concatenate([t.reshape(-1,3) for _,_,t in geo]);lo,hi=xyz.min(0),xyz.max(0)
        poly=MultiPoint(xyz[:,:2]).convex_hull
        assert poly.intersection(f.polygon()).area<1e-8 and lo[2]>=floor and hi[2]<ceil+1e-9
        for other,l,h in occupied:
            checks+=1
            assert not (poly.intersection(other).area>1e-8 and min(hi[2],h)>max(lo[2],l)+1e-8)
        occupied.append((poly,lo[2],hi[2]))
        assert all(np.isfinite(t).all() and np.all(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)>1e-9) for _,_,t in geo)
        out.append(dict(id=q.id,anchor=o.id,kind="balcony_glazed" if glazed else "balcony_open",
            facade=wall.id,triangles=sum(len(t) for _,_,t in geo),bounds=[lo.tolist(),hi.tolist()],geometry=geo))
    assert {m["anchor"] for m in out}==set(openings) and len(out)==len(openings)
    assert len(v.mass.shell.triangles())+sum(m["triangles"] for m in out)<=v.max_triangles
    return out,checks


def shader(old,cells,styles):
    def shade(sample,r,maps,mode,dense=False):
        col,em=old(sample,r,maps,"bare",False)
        u,z=sample.T;a=styles[r["building"]];k=r.get("accessory")
        if k in ("rail","glazed"):
            x,y=sample.T/np.asarray(r["span"])[:,None]
            if k=="rail":
                col[:]=srgb_decode(np.array([.25,.25,.23]))
            else:
                frame=(x<.045)|(x>.955)|(np.mod(x*3,1)<.05)|(y>.96)|(np.abs(y-.40)<.022)
                col[:]=srgb_decode(np.array([.56,.54,.49]))
                glass=y>.4
                col[glass]=srgb_decode(np.column_stack((.13+.28*y[glass],.22+.27*y[glass],.27+.29*y[glass])))
                col[frame]=srgb_decode(np.array([.76,.75,.70]))
            return col,em
        if r["role"]!="wall":return col,em
        role,var,local=tile.lookup(sample,r,cells);wall=role==0;h=r["floor_h"]
        # Shared material bank, coherent sections and construction rhythm.
        col[wall]*=np.asarray(a.accent)
        floor=np.floor(z/h);local_floor=z%h
        phase=(u-r["first_center"])/max(r["spacing"],.01)
        pier=np.abs((phase+.5)%1-.5)>.40
        base=z<.46;band=np.minimum(local_floor,h-local_floor)<.055
        if a.id in ("baltic_timber","nordic_cottage"):
            board=np.minimum(z%.19,.19-z%.19)<.012
            col[wall&board]*=.60
        if a.composition=="blind_end":
            col[wall&(band|pier)]*=.63
            col[wall]*=(.92+.08*np.sin(floor[wall]*2.1+np.floor(phase[wall])*1.7))[:,None]
        if a.composition=="sections":
            core=(np.floor(phase+.5)%4==3)
            col[wall&core]*=.70
        if a.composition=="ribbon":
            ribbon=(local_floor>r["sill"]-.08)&(local_floor<r["sill"]+r["window_height"]+.08)
            col[wall&ribbon]*=.34
        if a.id=="glass_grid":
            col[wall&(local_floor>.16)&(local_floor<h-.22)]*=.48
        if a.id in ("central_townhouse","brick_walkup","perimeter_block","masonry_office","civic_block"):
            col[wall&band]*=1.28
        col[wall&base]*=.46
        col[wall&(z>r["height"]-.16)&(z<r["height"])]*=1.24
        # Gable is deliberately blank above eave; no orphan roof-floor windows.
        col[wall]*=(.90+.065*np.sin(u*.39+z*.32)+.035*np.sin(u*1.71-z*.57))[wall,None]
        x,y=local.T;opening=(role==1)|(role==2)
        edge_x=np.minimum(x,1-x)*r["window_width"]
        edge_y=np.minimum(y,1-y)*np.where(role==2,r["door_height"],r["window_height"])
        mull=(np.abs(x-.5)<.023) if a.id!="glass_grid" else (np.abs(x-.5)<.012)
        glass=opening&(edge_x>.055)&(edge_y>.055)&~mull
        frame=np.array([.77,.75,.69]) if a.use!="office" else np.array([.37,.40,.41])
        col[opening]=srgb_decode(frame)
        variation=.025*np.sin(np.floor(phase+.5)*2.1+floor*1.7)
        col[glass]=srgb_decode(np.column_stack((.13+.23*y[glass]+variation[glass],.21+.23*y[glass]+variation[glass],.26+.26*y[glass]+variation[glass])))
        # Recess and sill shadow proxy only; no new window geometry.
        col[opening&(edge_y<.027)]*=.58
        door=role==3;col[door]=srgb_decode(np.array([.25,.20,.16]) if a.use=="house" else np.array([.24,.29,.28]))
        door_glass=door&(x>.14)&(x<.86)&(y>.44)&(y<.91)
        if a.use!="house":col[door_glass]=srgb_decode(np.array([.17,.29,.35]))
        if mode=="night":
            for i,name in enumerate(tile.VARIANTS):em[glass&(var==i)]=EMISSION[name]
        return np.clip(col,0,1),em
    return shade


def visibility(sample,r):
    if r.get("accessory")!="rail":return np.ones(len(sample),dtype=bool)
    u,z=sample.T;span=np.asarray(r["span"]);x=u/span[0];y=z/span[1]
    return (y<.075)|(y>.91)|(x<.035)|(x>.965)|(np.minimum(u%.18,.18-u%.18)<.017)


def render_views(v,uv,ids,table,cases,maps,cells):
    old=ref.shader;styles={c["building_id"]:c["variant"].archetype for c in cases}
    ref.shader=shader(old,cells,styles)
    grouped=defaultdict(list)
    for r in table:grouped[r["building"]].append(r["id"])
    def frame(vv,uu,ii,view,mode,resolution,center=None,fit_points=None):
        view=np.asarray(view,float);view/=np.linalg.norm(view)
        right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
        if center is None:center=(vv.min(0)+vv.max(0))/2
        p=(vv if fit_points is None else fit_points)-center;W,H=resolution
        extent=max(np.ptp(p@up),np.ptp(p@right)*H/W)
        # Fixed 78% framing at the same input dimensions/camera for A fixtures.
        ppm=640*.78/max(extent,1.)
        return ref.render(vv,uu,ii,table,maps,mode,center_override=center,ppm_override=ppm,
            view_override=view,visibility=visibility,two_sided=lambda r:r.get("accessory") in ("rail","glazed"),resolution=resolution)
    sheets={k:Image.new("RGB",(5*320,3*264),(242,243,243)) for k in ("a","b","front_a")}
    sheets["auto"]=Image.new("RGB",(3*320,2*264),(242,243,243))
    counts=Counter();views=[]
    def paste(sheet,key,im,label):
        i=counts[key];counts[key]+=1;cols=3 if key=="auto" else 5
        x=(i%cols)*320;y=(i//cols)*264
        pic=im.copy();pic.thumbnail((320,240));sheet.paste(pic,(x+(320-pic.width)//2,y))
        ImageDraw.Draw(sheet).text((x+5,y+242),label,fill=(22,28,30))
    try:
        for c in cases:
            a=c["variant"].archetype;key=c["id"];mask=np.isin(ids,grouped[c["building_id"]])
            vv,uu,ii=v[mask],uv[mask],ids[mask]
            # Use canonical host bounds, not accessory-dependent fit, for A comparisons.
            host=np.array(c["variant"].mass.shell.vertices)+np.asarray(c["offset"])
            center=(host.min(0)+host.max(0))/2
            host=np.array([[x,y,z] for x in (host.min(0)[0],host.max(0)[0]) for y in (host.min(0)[1],host.max(0)[1]) for z in (host.min(0)[2],host.max(0)[2])])
            im,dep=frame(vv,uu,ii,[1,-1.5,.85],"day",(1280,960),center,host)
            name=key+"_iso.png";im.save(OUT/name)
            group="auto" if key.startswith("auto_") else key[-1]
            label=(key[5:]+": " if group=="auto" else "")+a.id+" / "+c["roof"]
            paste(sheets[group],group,im,label)
            views.append(dict(file=name,foreground=int(np.isfinite(dep).sum()),building=c["building_id"]))
            if group=="a":
                fr,fd=frame(vv,uu,ii,[0,-1,0],"day",(1280,960),center,host)
                name=key+"_front.png";fr.save(OUT/name);paste(sheets["front_a"],"front_a",fr,label)
                views.append(dict(file=name,foreground=int(np.isfinite(fd).sum()),building=c["building_id"]))
            if key in ("panel_slab_a","brick_walkup_a","glass_grid_a"):
                night,nd=frame(vv,uu,ii,[1,-1.5,.85],"night",(1280,960),center,host)
                np.testing.assert_array_equal(dep,nd)
                name=key+"_night.png";night.save(OUT/name);views.append(dict(file=name,foreground=int(np.isfinite(nd).sum()),building=c["building_id"]))
        for key,sheet in sheets.items():sheet.save(OUT/("contact_"+key+".png"))
        overview,od=frame(v,uv,ids,[1,-1.6,1.3],"day",(1920,1080));overview.save(OUT/"neighborhood.png")
        views.append(dict(file="neighborhood.png",foreground=int(np.isfinite(od).sum())))
    finally:ref.shader=old
    return views


def main(decoder=None):
    OUT.mkdir(parents=True,exist_ok=True);cases=fixtures()
    groups,table,cells,records=tile.compile_scene(cases,audit_samples=1024)
    g2,t2,c2,r2=tile.compile_scene(list(reversed(cases)),audit_samples=1024)
    assert table==t2 and records==r2;np.testing.assert_array_equal(cells,c2)
    for role in groups:
        for k in groups[role]:np.testing.assert_array_equal(groups[role][k],g2[role][k])
    shell_triangles=sum(len(g["f"]) for g in groups.values());shell_records=len(table)
    by_id={r["building_id"]:r for r in records};all_modules=[];checks=0;plans=[]
    for c in sorted(cases,key=lambda q:q["id"]):
        v=c["variant"];mods,n=modules(c);checks+=n
        for m in mods:append_compact(groups,table,c,m)
        b=by_id[c["building_id"]];b["near_triangles"]=b["triangles"]+sum(m["triangles"] for m in mods)
        b["balcony_modules"]=len(mods);b["glazed_modules"]=sum(m["kind"]=="balcony_glazed" for m in mods)
        assert b["balcony_door_intents"]==len(mods) and b["near_triangles"]<=v.max_triangles
        assert b["plan_digest"]==v.plan.digest()
        all_modules.extend({k:x for k,x in m.items() if k!="geometry"} for m in mods)
        plans.append(dict(id=c["id"],footprint=b["footprint"],offset=c["offset"],source_constraints=c["source"],decision=v.decision(),architecture=asdict(v.plan)))
    assert table[:shell_records]==t2
    for role in groups:
        for k,arr in g2[role].items():np.testing.assert_array_equal(groups[role][k][:len(arr)],arr)
    path=OUT/"tile.glb";tile.export_glb(groups,path);v,uv,ids,_=tile.read_glb(path,groups,table)
    assert len(v)//3==sum(b["near_triangles"] for b in records)
    dump(OUT/"parameters.json",table);dump(OUT/"plans.json",plans);dump(OUT/"placements.json",all_modules)
    (OUT/"facade_cells.bin").write_bytes(cells.tobytes())
    audit=new_float32_audit();samples=pack_table(table,cells,OUT/"facade_params.bin",audit)
    cpp=None
    if decoder:
        run=subprocess.run([str(decoder),str(OUT/"facade_params.bin"),str(OUT/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True)
        cpp=json.loads(run.stdout);assert cpp["records"]==len(table) and cpp["cells"]==len(cells)
        dump(OUT/"cpp_abi_check.json",cpp)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in {r["family"] for r in table}}
    views=render_views(v,uv,ids,table,cases,maps,cells)
    report=dict(schema="pcg.research.archetypes.v1",buildings=len(cases),archetypes=len({c["variant"].archetype.id for c in cases}),
        auto_selected=sum(c["id"].startswith("auto_") for c in cases),roof_counts=dict(Counter(b["roof"] for b in records)),
        shell_triangles=shell_triangles,near_triangles=len(v)//3,opening_triangles_added=0,balcony_modules=len(all_modules),
        max_building_triangles=max(b["near_triangles"] for b in records),surface_records=len(table),cell_bytes=len(cells),
        clearance_pair_checks=checks,canonical_lookup_samples=1024*sum(b["facades"] for b in records),float32_abi_samples=samples,
        float32_audit=audit,cpp_abi=cpp,glb_roundtrip=True,input_order_invariant=True,roof_iou_min=min(b["roof_iou"] for b in records),
        buildings_detail=records,views=views,limitations=["Synthetic regional priors; not calibrated or GIS ground truth.",
        "Stage25 architectural lane: CPU diffuse+emission and alpha visibility proxy, not the preceding GPU PBR/sidecar pipeline.",
        "Host+external balconies budget only; earlier roof equipment, widened paired modules, POM, private lobby and native adapter not integrated in this lane.",
        "No native GPU timing, measured LOD threshold or general concave pitched-roof support."])
    assert report["buildings"]==36 and report["archetypes"]==15 and set(report["roof_counts"])=={"flat","gable","hip"}
    dump(OUT/"report.json",report)
    dump(OUT/"bundle.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="bundle.json"})
    print("STAGE25",json.dumps({k:v for k,v in report.items() if k not in ("buildings_detail","views","float32_audit","limitations")}),flush=True)


if __name__=="__main__":main(sys.argv[1] if len(sys.argv)>1 else None)
