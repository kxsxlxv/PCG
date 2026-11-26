"""SDK-free reference-driven facade experiment; never a production adapter.
Reuse canonical bay/asset plans and GLB/PCGF encoders. All added triangles
represent balcony envelopes/slabs or equipment, not window or door meshes.
"""
from pathlib import Path
from dataclasses import replace
from collections import Counter,defaultdict
import base64,copy,hashlib,io,json,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import Polygon
from pcg.footprint import Footprint
from pcg.architecture import (plan_facades,PROFILES,BALCONY_DEPTH,BALCONY_BASE,
                              BALCONY_SLAB,BALCONY_HEIGHT)
from pcg.assets import plan_assets,asset_triangles
from pcg.materials import MATERIALS,texture_maps
from pcg.occupancy import EMISSION
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table,LAYERS,opening_rectangles
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_roof_budgets as roof

OUT=Path("artifacts/zero-facade-architecture")
LANES=(("plain",0),("zones",0),("minimal",64),("medium",128),("balanced",192),("near",320))
STYLES=("mixed_use","panel_block","brick_stack")
FAMILIES={"mixed_use":"plaster_cream","panel_block":"concrete_panel","brick_stack":"red_brick"}
def fixtures():
    rows=[("mixed_use",30,12,6,"flat",0),("panel_block",24,12,7,"flat",0),
          ("brick_stack",30,12,5,"gable",2.5)]
    return [dict(id=k,building_id="facade:"+k,outer=[(0,0),(w,0),(w,d),(0,d)],holes=[],
                 profile="apartments",floors=f,eave_m=3.*f,roof=r,rise=rise,
                 party_edges=[],offset=[i*46.,0.,0.])
            for i,(k,w,d,f,r,rise) in enumerate(rows)]

def styled_plan(f,h,levels,profile,**kwargs):
    p=plan_facades(f,h,levels,profile,**kwargs)
    style=f.building_id.split(":")[-1]
    assert style in STYLES
    longest=max(a.length for a in p.facades)
    fs=[];rejected=list(p.balcony_fallbacks)
    for a in p.facades:
        long=a.length>=.9*longest
        # Preserve canonical slot IDs, aperture alignment and fit. A single
        # research plan is fixed before compiling all geometry budgets.
        candidates=sorted({round(o.u,8) for o in a.openings if o.role=="balcony_door"})
        chosen=set(candidates[1::3] if style=="mixed_use" else candidates[::3])
        ops=[]
        for o in a.openings:
            if style=="panel_block" and not long and o.role!="door":
                rejected.append((o.id,"research_blind_end"));continue
            if o.role=="balcony_door" and (not long or round(o.u,8) not in chosen or
                    (style=="mixed_use" and o.floor%2==0)):
                rejected.append((o.id,"research_architecture_window_instead_of_balcony"))
                o=replace(o,role="window",z=p.floor_levels[o.floor]+profile.sill,height=profile.window_height)
            ops.append(o)
        fs.append(replace(a,openings=tuple(ops)))
    return replace(p,facades=tuple(fs),balcony_fallbacks=tuple(rejected))

def box_local(w,d,h,x=0.,y=0.,z=0.):
    return roof.box_mesh(x,y,w,d,h,z)

def panel_quad(a,b,z0,z1):
    # Local coordinates: X tangent, Y negative outside, Z up.
    # Outward-oriented open proxy skin, not closed solid glazing.
    p=np.array([[*a,z0],[*b,z0],[*b,z1],[*a,z1]],float)
    return p[[[0,1,2],[0,2,3]]]

def balcony(recipe,placement,style,opening):
    w=recipe.width;d=BALCONY_DEPTH
    # Reuse canonical slab triangles, not the window/frame assets.
    slab=np.array([tri for role,tri in asset_triangles(recipe,1) if role=="balcony_slab"])
    assert len(slab)==12
    glazed=ref.seed(opening.id)%5 < (1 if style=="mixed_use" else 4)
    high=min(2.55,opening.height+.20) if glazed else BALCONY_HEIGHT
    a=(-w/2,0.);b=(-w/2,-d);c=(w/2,-d);e=(w/2,0.)
    parts=[("slab","concrete_panel",slab)]
    mode="glazed" if glazed else "rail"
    for q,t in ((a,b),(b,c),(c,e)):
        parts.append((mode,"painted_rail",panel_quad(q,t,BALCONY_SLAB,high)))
    # Enclosure cover is an architectural volume, not an aperture mesh.
    if glazed:
        cap=np.array([[*b,high],[*c,high],[*e,high],[*a,high]])
        parts.append(("slab","concrete_panel",cap[[[0,1,2],[0,2,3]]]))
    return [(k,m,np.array([[placement.point(p) for p in tri] for tri in t])) for k,m,t in parts]

def world_box(fac,u,z,w,d,h):
    p=box_local(w,d,h,y=-d/2)
    return np.array([[fac.point(u+x,z+zz,depth=y) for x,y,zz in tri] for tri in p])

def plan_modules(c):
    f=Footprint(c["building_id"],c["outer"],c["holes"])
    p=styled_plan(f,c["eave_m"],c["floors"],PROFILES[c["profile"]],party_edges=())
    aplan=plan_assets(f,p);recipes={r.id:r for r in aplan.recipes}
    facades={a.id:a for a in p.facades};ops={o.id:o for a in p.facades for o in a.openings}
    modules=[];occupied=[];rejected=[]
    def add(mid,kind,fac,anchor,parts):
        verts=np.concatenate([t.reshape(-1,3) for _,_,t in parts])
        assert np.isfinite(verts).all()
        for _,_,t in parts:
            assert np.all(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)>1e-8)
        lo,hi=verts.min(0),verts.max(0)
        # Footprint must be outside host except intentional zero-width contact.
        from shapely.geometry import MultiPoint
        poly=MultiPoint(verts[:,:2]).convex_hull
        assert poly.area>0
        if poly.intersection(f.polygon()).area>1e-8 or lo[2]<0 or hi[2]>c["eave_m"]+1e-8:
            rejected.append((mid,"host_fit"));return
        if any(poly.intersection(q).area>1e-9 and min(hi[2],h)>max(lo[2],l)+1e-8 for q,l,h in occupied):
            rejected.append((mid,"accessory_overlap"));return
        occupied.append((poly,lo[2],hi[2]))
        modules.append(dict(id=mid,kind=kind,facade=fac.id,anchor=anchor,triangles=sum(len(t) for _,_,t in parts),
                            bounds=[lo.tolist(),hi.tolist()],geometry=parts))
    # Whole columns are ordered together; canonical anchor IDs survive all LODs.
    placements=[q for q in aplan.placements if recipes[q.asset_id].family=="balcony"]
    placements.sort(key=lambda q:(facades[q.facade_id].outward[1]>=0,q.facade_id,round(ops[q.opening_id].u,6),ops[q.opening_id].floor))
    for q in placements:
        o=ops[q.opening_id];a=facades[q.facade_id];r=recipes[q.asset_id]
        parts=balcony(r,q,c["id"],o)
        add(q.id,"balcony_glazed" if any(k=="glazed" for k,_,_ in parts) else "balcony_open",a,o.id,parts)
    balconies=list(modules)
    assert balconies and not rejected, (c["id"],rejected)
    # Equipment is tested against the real canonical apertures in facade UV.
    def equipment(a,key,kind,u,z,w,d,h,mat):
        if u-w/2<.05 or u+w/2>a.length-.05 or z<0 or z+h>c["eave_m"]:
            return
        if any(min(u+w/2,o.u+o.width)>max(u-w/2,o.u)+1e-6 and
               min(z+h,o.z+o.height)>max(z,o.z)+1e-6 for o in a.openings):
            return
        add(a.id+":"+key,kind,a,key,[(kind,mat,world_box(a,u,z,w,d,h))])
    for a in p.facades:
        if a.party or a.courtyard:continue
        if a.outward[1]<0:
            for o in a.openings:
                if o.role=="window" and o.z>.6 and ref.seed(o.id)%7==0:
                    equipment(a,o.id+":ac","ac",o.u+o.width/2,o.z-.52,.74,.28,.43,"painted_trim")
            if c["id"]=="mixed_use":
                for o in a.openings:
                    if o.role=="door":
                        equipment(a,"entry_canopy","canopy",o.u+o.width/2,o.z+o.height+.10,o.width+.6,.85,.18,"metal_roof_red")
            if c["id"]=="brick_stack":
                equipment(a,"drain","drain",.28,.10,.10,.13,c["eave_m"]-.20,"painted_rail")
    extra=modules[len(balconies):]
    # First compact stack, equipment, then remaining columns. Monotone
    # prefix selection keeps stable geometry and indices as budget grows.
    first=balconies[0];firstcol=ops[first["anchor"]].u
    stack=[m for m in balconies if m["facade"]==first["facade"] and abs(ops[m["anchor"]].u-firstcol)<1e-8]
    rest=[m for m in balconies if m["id"] not in {q["id"] for q in stack}]
    order=stack[:2]+extra[:3]+stack[2:]+rest+extra[3:]
    assert len({m["id"] for m in order})==len(order)
    return p,order,rejected

def append(groups,table,c,m):
    # Existing non-wall primitive is a neutral ABI bucket for architectural
    # accessories; semantic role and envelope shading live in explicit sidecar.
    # No change to PCGF v1 layout or existing role semantics of canonical shell.
    g=groups["roof"];offset=np.asarray(c["offset"])
    for j,(kind,family,tris) in enumerate(m["geometry"]):
        for i,tri in enumerate(tris):
            sid=m["id"]+":"+str(j)+":"+str(i);rid=len(table);base=len(g["v"])
            n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
            tangent=np.cross([0.,0.,1.],n)
            if np.linalg.norm(tangent)<1e-10:tangent=np.array([1.,0.,0.])-n*n[0]
            tangent/=np.linalg.norm(tangent);bitangent=np.cross(n,tangent)
            # UV anchored to the full part, never to its individual triangle.
            allv=tris.reshape(-1,3);xy=np.column_stack((allv@tangent,allv@bitangent))
            uv=np.column_stack((tri@tangent,tri@bitangent))-xy.min(0)
            span=np.maximum(xy.max(0)-xy.min(0),1e-6)
            b=ref.pack([rid])[0]
            for k,val in (("v",tri+offset),("n",np.tile(n,(3,1))),("uv0",uv/MATERIALS[family].repeat_m),
                          ("uv1",uv),("color",np.tile([b[0],b[1],0,255],(3,1))),("f",[[base,base+1,base+2]])):
                g[k]=np.concatenate((g[k],np.asarray(val,dtype=g[k].dtype)))
            table.append(dict(id=rid,building=c["building_id"],surface=sid,role="roof",family=family,
                              seed=ref.seed(sid),height=c["eave_m"],floors=c["floors"],
                              accessory=kind,span=span.tolist(),placement=m["id"]))

def render_shader(old,cells,styles,zoned,joint_masks=None):
    def shade(sample,r,maps,mode,dense=False):
        col,em=old(sample,r,maps,"bare",False)
        u,z=sample.T
        if "accessory" in r:
            k=r["accessory"];x,y=sample.T/np.asarray(r["span"])[:,None]
            if k in ("rail","glazed"):
                color=np.array([.37,.18,.14]) if styles[r["building"]]=="panel_block" else np.array([.26,.22,.19])
                col[:]=srgb_decode(color)
                slats=(np.mod(x*18,1)<.18);col[slats]*=.75
                if k=="glazed":
                    glass=y>.39
                    frame=(np.mod(x*4,1)<.065)|(x>.97)|(y>.96)|(np.abs(y-.41)<.022)
                    col[glass]=srgb_decode(np.array([.23,.34,.39]))
                    col[glass&frame]=srgb_decode(np.array([.80,.79,.75]))
            elif k=="ac":
                disk=(x-.47)**2+(y-.50)**2<.32**2
                col[disk]=srgb_decode(np.array([.20,.21,.21]))
                col[disk&(np.mod(y*20,1)<.25)]*=1.5
            return col,em
        if r["role"]!="wall":return col,em
        role,var,loc=tile.lookup(sample,r,cells);wall=role==0
        style=styles[r["building"]]
        if zoned:
            base=maps[FAMILIES[style]];repeat=MATERIALS[FAMILIES[style]].repeat_m
            ix=np.floor((u/repeat%1)*base.shape[1]).astype(int)%base.shape[1]
            iy=np.floor((-z/repeat%1)*base.shape[0]).astype(int)%base.shape[0]
            col[wall]=srgb_decode(base[iy[wall],ix[wall]])
            h=r["floor_h"];floor=np.floor(z/h).astype(int);bay=np.floor(u/2.8).astype(int)
            # Coherent panel variation and permanent joint-localized wear.
            n=(np.sin(u*2.6+z*.47)+np.sin(u*.73-z*3.1))*.5
            if style=="panel_block":
                joints=(np.minimum(z%h,h-z%h)<.045)|(np.minimum(u%2.8,2.8-u%2.8)<.035)
                if joint_masks is not None:joints=joint_masks(r,sample)
                col[wall]*=(.94+.055*np.sin(floor[wall]*2.3+bay[wall]*1.7))[:,None]
                col[wall&joints]*=(.55+.20*n[wall&joints])[:,None]
            if style=="mixed_use":
                col[wall&(z<h)]*=.49
                col[wall&(np.abs(z-h)<.18)]*=1.40
                col[wall&(np.abs(z-r["height"]*.53)<.10)]*=1.22
            col[wall&(z<.45)]*=.55
            col[wall&(z>r["height"]-.25)]*=1.18
            col[wall]*=(.94+.06*n[wall])[:,None]
        x,y=loc.T;opening=(role==1)|(role==2)
        glass=opening&(x>.08)&(x<.92)&(y>.08)&(y<.92)&(np.abs(x-.5)>.035)
        col[opening]=srgb_decode(np.array([.79,.77,.73]))
        col[glass]=srgb_decode(np.column_stack((.16+.13*y[glass],.24+.14*y[glass],.30+.14*y[glass])))
        col[role==3]=srgb_decode(np.array([.32,.22,.14]))
        if mode=="night":
            for i,name in enumerate(tile.VARIANTS):em[glass&(var==i)]=EMISSION[name]
        return col,em
    return shade

def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True);cases=fixtures()
    styles={c["building_id"]:c["id"] for c in cases}
    original=tile.plan_facades
    try:
        tile.plan_facades=styled_plan
        g0,t0,cells,owners=tile.compile_scene(cases,audit_samples=2048)
    finally:tile.plan_facades=original
    plans={c["id"]:plan_modules(c) for c in cases}
    for c in cases:
        p,a,b=plans[c["id"]];q,aa,bb=plan_modules(c)
        assert p.digest()==q.digest() and b==bb
        assert [m["id"] for m in a]==[m["id"] for m in aa]
        for m,n in zip(a,aa):
            for (_,_,x),(_,_,y) in zip(m["geometry"],n["geometry"]):np.testing.assert_array_equal(x,y)
        assert p.digest()==next(o["plan_digest"] for o in owners if o["building_id"]==p.building_id)
    for r in t0:
        if r["role"]=="wall":
            a=next(a for p,_,_ in plans.values() for a in p.facades if a.id==r["facade_id"])
            assert len(opening_rectangles(r,cells))==len(a.openings)
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    old=ref.shader;reports=[];previous=set();baseline=None
    cameras=[((61,6,10),4.3,(.35,-1,.45)),((61,6,10),4.3,(.3,-1,1.8))]
    for c in cases:
        w=c["outer"][1][0];h=c["eave_m"]
        cameras.append(((c["offset"][0]+w/2,6,h/2),12.,(.23,-1,.38)))
    sheet=Image.new("RGB",(640*len(LANES),640*len(cameras)+32),(230,230,230))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLB reload | fixed cameras | CPU diffuse | facade zones and nested accessory budgets",fill=(0,0,0))
    try:
        for col,(name,budget) in enumerate(LANES):
            d=OUT/name;d.mkdir(exist_ok=True)
            g=copy.deepcopy(g0);t=copy.deepcopy(t0);selected=[];counts={}
            for c in cases:
                modules=roof.select(plans[c["id"]][1],budget)
                counts[c["id"]]=sum(m["triangles"] for m in modules)
                assert counts[c["id"]]<=budget
                for m in modules:
                    append(g,t,c,m);selected.append({k:v for k,v in m.items() if k!="geometry"})
            keys={m["id"] for m in selected};assert previous<=keys;previous=keys
            assert t[:len(t0)]==t0
            for role in tile.GROUPS:
                for k,a in g0[role].items():np.testing.assert_array_equal(g[role][k][:len(a)],a)
            mesh=d/"tile.glb";tile.export_glb(g,mesh);v,uv,ids,_=tile.read_glb(mesh,g,t)
            shell_tri=sum(len(a["f"]) for a in g0.values())
            assert len(v)//3==shell_tri+sum(counts.values())
            (d/"facade_cells.bin").write_bytes(cells.tobytes())
            samples=pack_table(t,cells,d/"facade_params.bin")
            result=json.loads(subprocess.run([decoder,str(d/"facade_params.bin"),str(d/"facade_cells.bin"),str(len(LAYERS))],check=True,capture_output=True,text=True).stdout)
            assert result["records"]==len(t) and result["cells"]==len(cells)
            (d/"parameters.json").write_text(json.dumps(t,indent=2))
            (d/"placements.json").write_text(json.dumps(selected,indent=2))
            ref.shader=render_shader(old,cells,styles,name!="plain")
            depths=[]
            for row,(center,ppm,view) in enumerate(cameras):
                im,dep=ref.render(v,uv,ids,t,maps,"day",center_override=center,ppm_override=ppm,view_override=view)
                im.save(d/f"view_{row}.png");depths.append(dep)
                ImageDraw.Draw(im).text((8,8),f"{name} | +{budget} max/building | {len(v)//3} tris",fill=(0,0,0))
                sheet.paste(im,(col*640,row*640+32))
            if baseline is None:baseline=depths
            change=[]
            for a,b in zip(baseline,depths):
                common=np.isfinite(a)
                assert np.all(~common|np.isfinite(b))
                assert np.all(b[common]>=a[common]-1e-5)
                if budget==0:np.testing.assert_array_equal(a,b)
                change.append(int(np.sum(common&(b>a+1e-5))))
            # Check canonical per-opening emission, independent of accessory occlusion.
            for r in t0:
                if r["role"]!="wall":continue
                sample=np.random.default_rng(r["seed"]).random((512,2))*[r["length"],r["height"]]
                _,e0=render_shader(old,cells,styles,False)(sample,r,maps,"night")
                _,e1=ref.shader(sample,r,maps,"night")
                np.testing.assert_array_equal(e0,e1)
            reports.append(dict(name=name,budget=budget,triangles=len(v)//3,per_building_added=counts,
                modules=len(selected),kinds=dict(Counter(m["kind"] for m in selected)),bytes=mesh.stat().st_size,
                abi_samples=samples,cpp_abi=True,depth_changed_pixels=change,opening_triangles_added=0,
                canonical_opening_cells_sha256=hashlib.sha256(cells.tobytes()).hexdigest()))
        assert reports[-1]["modules"]>0 and sum(reports[-1]["depth_changed_pixels"])>0
        assert reports[-1]["kinds"].get("balcony_glazed",0)>0 and reports[-1]["kinds"].get("balcony_open",0)>0
    finally:ref.shader=old
    sheet.save(OUT/"contact.png")
    report={"schema":"pcg.research.facade-architecture.v1","buildings":owners,"lanes":reports,
        "planned":{k:dict(modules=len(v[1]),rejections=v[2]) for k,v in plans.items()},
        "limitations":["synthetic rectangular fixtures, CPU diffuse only; no native or hardware timing",
        "opaque balcony envelope proxy with shader glazing illusion, not transparent glass or collision",
        "budget may omit canonical balcony anchors; visual research only, not safe physical balcony access",
        "mixed-use ground-floor material band only, no new storefront aperture contract",
        "panel joints and accessory shading diagnostic sidecar only, not native PCGF extension",
        "roof geometry tested separately; no snow or icicles, no weather/season overlays"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("FACADE_ARCH_REPORT "+json.dumps(report),flush=True)
    # Individual close-up sheet at legible preview scale.
    thumb=Image.new("RGB",(640*3,640*3))
    for row in range(3):
        for col,name in enumerate(("plain","zones","near")):
            im=Image.open(OUT/name/f"view_{row+2}.png")
            ImageDraw.Draw(im).text((8,8),name,fill=(0,0,0));thumb.paste(im,(col*640,row*640))
    thumb.save(OUT/"close_contact.png");thumb.thumbnail((1200,1200))
    b=io.BytesIO();thumb.save(b,format="JPEG",quality=73,optimize=True)
    s=base64.b64encode(b.getvalue()).decode()
    for i in range(0,len(s),4000):print(f"FACADE_ARCH_PREVIEW_{i//4000:03d} "+s[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
