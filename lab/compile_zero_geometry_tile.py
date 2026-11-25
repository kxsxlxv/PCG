"""Research-only neutral tile compiler. Reuse canonical shell, architecture and
occupancy plans. One GLB node, one mesh, three role primitives, no opening geometry.
Custom runtime material and native SDK import are NOT implemented.
"""
from pathlib import Path
from dataclasses import asdict
import base64
import hashlib
import io
import json
import struct
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pcg.footprint import Footprint
from pcg.massing import build_massing
from pcg.architecture import PROFILES, plan_facades, BALCONY_THRESHOLD
from pcg._grammar_bays import window_offsets
from pcg.materials import MATERIALS, texture_maps, resolve_material
from pcg.occupancy import plan_night, EMISSION
from pcg.far_emission import srgb_decode
import render_zero_geometry_facades as ref

OUT=Path("artifacts/zero-tile")
ROLE_CODES={"window":1,"balcony_door":2,"door":3}
VARIANTS=("off","warm","neutral")
ORIGIN=np.array([700000.,4800000.,125.],dtype=np.float64) # synthetic projected frame
GROUPS=("wall","roof","base")

def lookup(uv,r,cells):
    """Bounded O(1) regular-cell lookup plus one ground-door exception."""
    u,z=uv.T
    n=len(u)
    role=np.zeros(n,dtype=np.uint8)
    variant=np.zeros(n,dtype=np.uint8)
    local=np.zeros((n,2),dtype=np.float64)
    if r["bays"]:
        f=np.floor(z/r["floor_h"]).astype(np.int64)
        b=np.floor((u-r["first_center"])/r["spacing"]+.5).astype(np.int64)
        valid=(f>=0)&(f<r["floors"])&(b>=0)&(b<r["bays"])
        ii=np.flatnonzero(valid)
        if len(ii):
            code=cells[r["cell_start"]+f[ii]*r["bays"]+b[ii]]
            kind=code&3
            center=r["first_center"]+b[ii]*r["spacing"]
            bottom=f[ii]*r["floor_h"]+np.where(kind==2,r["balcony_threshold"],r["sill"])
            height=np.where(kind==2,r["door_height"],r["window_height"])
            x=(u[ii]-center)/r["window_width"]+.5
            y=(z[ii]-bottom)/height
            ok=(kind>0)&(x>=0)&(x<1)&(y>=0)&(y<1)
            ii=ii[ok]
            role[ii]=kind[ok];variant[ii]=code[ok]>>2
            local[ii]=np.column_stack((x[ok],y[ok]))
    if r["door"] is not None:
        a,b,w,h=r["door"]
        x=(u-a)/w;y=(z-b)/h
        ok=(x>=0)&(x<1)&(y>=0)&(y<1)
        assert not np.any(role[ok]),"door overlaps a planned window"
        role[ok]=3;local[ok]=np.column_stack((x[ok],y[ok]))
    return role,variant,local

def compile_record(facade,plan,h,cells,intents):
    p=plan.profile;floors=len(plan.floor_levels)-1
    centers=[] if facade.party else window_offsets(facade.length,p.window_width,p.spacing,p.margin)
    r={"bays":len(centers),"first_center":centers[0] if centers else 0.,
       "spacing":p.spacing,"window_width":p.window_width,"window_height":p.window_height,
       "sill":p.sill,"door_height":p.door_height,"balcony_threshold":BALCONY_THRESHOLD,
       "floor_h":h/floors,"floors":floors,"height":h,"length":facade.length,
       "cell_start":len(cells),"door":None,"facade_id":facade.id,"party":facade.party,
       "courtyard":facade.courtyard,"seed":ref.seed(facade.id)}
    codes=np.zeros((floors,len(centers)),dtype=np.uint8)
    for o in facade.openings:
        if o.role=="door":
            assert r["door"] is None
            r["door"]=[o.u,o.z,o.width,o.height]
            continue
        b=round((o.u+o.width*.5-r["first_center"])/p.spacing)
        assert 0<=b<len(centers) and 0<=o.floor<floors
        assert codes[o.floor,b]==0
        np.testing.assert_allclose(o.u+o.width*.5,centers[b],atol=1e-9)
        np.testing.assert_allclose(o.width,p.window_width,atol=1e-9)
        expected_z=plan.floor_levels[o.floor]+(BALCONY_THRESHOLD if o.role=="balcony_door" else p.sill)
        expected_h=p.door_height if o.role=="balcony_door" else p.window_height
        np.testing.assert_allclose([o.z,o.height],[expected_z,expected_h],atol=1e-9)
        codes[o.floor,b]=ROLE_CODES[o.role]|(VARIANTS.index(intents[o.id].variant)<<2)
    cells.extend(codes.ravel().tolist())
    # Compare direct canonical rectangle membership to packed lookup,
    # independent of render and all stable IDs.
    rng=np.random.default_rng(r["seed"])
    sample=rng.random((4096,2))*[facade.length,h]
    expected=np.zeros(len(sample),np.uint8);ev=np.zeros(len(sample),np.uint8)
    for o in facade.openings:
        mask=(sample[:,0]>=o.u)&(sample[:,0]<o.u+o.width)&(sample[:,1]>=o.z)&(sample[:,1]<o.z+o.height)
        assert not np.any(expected[mask])
        expected[mask]=ROLE_CODES[o.role]
        if o.role!="door":ev[mask]=VARIANTS.index(intents[o.id].variant)
    actual,av,_=lookup(sample,r,np.asarray(cells,dtype=np.uint8))
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(av,ev)
    return r

def compile_scene(cases):
    table=[];cells=[];records=[];groups={role:{k:[] for k in ("v","n","uv0","uv1","color","f")} for role in GROUPS}
    for i,c in enumerate(sorted(cases,key=lambda a:a["id"])):
        f=Footprint("fixture:"+c["id"],c["outer"],c.get("holes",[]))
        profile=PROFILES[("house","apartments","office")[i%3]]
        floors=2+i%3;h=floors*(3.2 if profile.id=="office" else 3.)
        mass=build_massing(f,h,c["roof"],c.get("rise",0.))
        assert bool(mass.fallback_reason)==bool(c.get("fallback",False)),c["id"]
        shell=mass.shell
        v=np.asarray(shell.vertices,dtype=np.float64)
        welded=trimesh.Trimesh(vertices=v,faces=[t for _,t in shell.triangles()],process=False)
        assert welded.is_watertight and welded.is_winding_consistent and welded.volume>0
        party=((0,1),) if c["id"]=="L" else ()
        plan=plan_facades(f,h,floors,profile,party_edges=party)
        intents={x.opening_id:x for x in plan_night(plan)}
        offset=np.array([(i%4)*24.,(i//4)*24.,0.])
        matched=set();roof=[];before=sum(len(g["f"]) for g in groups.values())
        for s in shell.surfaces:
            p=v[list(s.vertices)]
            tris=np.array([(0,j,j+1) for j in range(1,len(p)-1)],dtype=np.int64)
            n=np.cross(p[1]-p[0],p[2]-p[0]);n/=np.linalg.norm(n)
            assert np.isfinite(n).all()
            assert np.all(np.cross(p[tris[:,1]]-p[tris[:,0]],p[tris[:,2]]-p[tris[:,0]])@n>1e-9)
            r={"id":len(table),"building":f.building_id,"surface":s.id,"role":s.role,
               "seed":ref.seed(s.id),"family":resolve_material(profile.id,s.role).id,
               "height":h,"floors":floors}
            if s.role=="wall":
                found=[a for a in plan.facades if np.allclose(p[0,:2],a.start,atol=1e-9,rtol=0) and np.allclose(p[1,:2],a.end,atol=1e-9,rtol=0)]
                assert len(found)==1,(c["id"],s.id)
                facade=found[0];assert facade.id not in matched;matched.add(facade.id)
                r.update(compile_record(facade,plan,h,cells,intents))
                direction=(np.asarray(facade.end)-facade.start)/facade.length
                layout=np.column_stack(((p[:,:2]-facade.start)@direction,p[:,2]))
            else:
                if s.role=="roof":
                    if mass.actual_roof!="flat":r["family"]="ceramic_roof"
                    roof.extend(Polygon(p[t,:2]) for t in tris)
                tangent=np.cross([0.,0.,1.],n)
                if np.linalg.norm(tangent)<1e-10:tangent=np.array([1.,0.,0.])-n*n[0]
                tangent/=np.linalg.norm(tangent)
                bitangent=np.cross(n,tangent)
                layout=np.column_stack((p@tangent,p@bitangent))
            table.append(r)
            g=groups[s.role];base=len(g["v"])
            g["v"].extend(p+offset);g["n"].extend(np.tile(n,(len(p),1)))
            g["uv1"].extend(layout);g["uv0"].extend(layout/MATERIALS[r["family"]].repeat_m)
            byte=ref.pack([r["id"]])[0]
            g["color"].extend(np.tile([int(byte[0]),int(byte[1]),0,255],(len(p),1)))
            g["f"].extend(tris+base)
        assert matched=={a.id for a in plan.facades}
        coverage=unary_union(roof)
        assert coverage.symmetric_difference(f.polygon()).area<1e-7,c["id"]
        assert abs(sum(a.area for a in roof)-coverage.area)<1e-7
        actual=sum(len(g["f"]) for g in groups.values())-before
        assert actual==len(shell.triangles())
        records.append({"building_id":f.building_id,"footprint":asdict(f),"roof":mass.actual_roof,
                        "fallback_reason":mass.fallback_reason,"offset_local":offset.tolist(),
                        "triangles":actual,"facades":len(plan.facades),"openings":sum(len(a.openings) for a in plan.facades),
                        "balcony_door_intents":sum(o.role=="balcony_door" for a in plan.facades for o in a.openings),
                        "plan_digest":plan.digest(),"roof_iou":coverage.intersection(f.polygon()).area/coverage.union(f.polygon()).area})
    arrays={role:{k:np.asarray(a,dtype=np.uint8 if k=="color" else np.uint32 if k=="f" else np.float32) for k,a in g.items()} for role,g in groups.items()}
    return arrays,table,np.asarray(cells,dtype=np.uint8),records

def y_up(v):
    return np.column_stack((v[:,0],v[:,2],-v[:,1])).astype(np.float32)
def z_up(v):
    return np.column_stack((v[:,0],-v[:,2],v[:,1])).astype(np.float32)

def export_glb(groups,path):
    scene=trimesh.Scene()
    for i,role in enumerate(GROUPS):
        g=groups[role]
        visual=trimesh.visual.texture.TextureVisuals(uv=g["uv0"],
            material=trimesh.visual.material.PBRMaterial(name="pcg_"+role,baseColorFactor=[.45+.1*i,.5,.5,1.]))
        visual.vertex_attributes["color"]=g["color"]
        m=trimesh.Trimesh(vertices=y_up(g["v"]),faces=g["f"],vertex_normals=y_up(g["n"]),
             process=False,visual=visual,vertex_attributes={"_PCG_LAYOUT":g["uv1"]})
        scene.add_geometry(m,geom_name=role,node_name=role)
    def post(tree):
        primitives=[]
        for mesh in tree["meshes"]:
            for p in mesh["primitives"]:
                p["attributes"]["TEXCOORD_1"]=p["attributes"].pop("_PCG_LAYOUT")
                p["extras"]={"pcg_role":mesh["name"]}
                primitives.append(p)
        tree["meshes"]=[{"name":"pcg_tile","primitives":primitives}]
        tree["nodes"]=[{"name":"pcg_tile","mesh":0}]
        tree["scenes"]=[{"nodes":[0]}];tree["scene"]=0
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene,include_normals=True,tree_postprocessor=post))

def read_glb(path,groups,table):
    """Independent raw accessor audit: generic viewers may discard UV1."""
    b=path.read_bytes()
    magic,version,total=struct.unpack_from("<4sII",b,0)
    assert magic==b"glTF" and version==2 and total==len(b)
    n,kind=struct.unpack_from("<I4s",b,12);assert kind==b"JSON"
    tree=json.loads(b[20:20+n])
    bn,bkind=struct.unpack_from("<I4s",b,20+n);assert bkind==b"BIN\x00"
    blob=b[28+n:28+n+bn];assert len(blob)==bn
    assert len(tree["nodes"])==len(tree["meshes"])==1
    assert not tree.get("images"),"unique image allocation"
    def read(acc):
        a=tree["accessors"][acc];view=tree["bufferViews"][a["bufferView"]]
        dt=np.dtype({5121:"u1",5123:"<u2",5125:"<u4",5126:"<f4"}[a["componentType"]])
        width={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[a["type"]]
        assert view.get("byteStride",width*dt.itemsize)==width*dt.itemsize
        start=view.get("byteOffset",0)+a.get("byteOffset",0)
        assert a.get("byteOffset",0)+a["count"]*width*dt.itemsize<=view["byteLength"]
        return np.frombuffer(blob,dtype=dt,count=a["count"]*width,offset=start).reshape(-1,width)
    out=[];uv=[];ids=[];roles=[]
    pp=tree["meshes"][0]["primitives"];assert len(pp)==len(GROUPS)
    for p in pp:
        role=p["extras"]["pcg_role"];g=groups[role];a=p["attributes"]
        pos=z_up(read(a["POSITION"]));normal=z_up(read(a["NORMAL"]))
        uv0=read(a["TEXCOORD_0"]).copy();uv0[:,1]=1-uv0[:,1]
        layout=read(a["TEXCOORD_1"]);color=read(a["COLOR_0"])
        assert tree["accessors"][a["COLOR_0"]]["normalized"] is True
        faces=read(p["indices"]).reshape(-1,3)
        np.testing.assert_array_equal(pos,g["v"]);np.testing.assert_allclose(normal,g["n"],atol=1e-6)
        np.testing.assert_allclose(uv0,g["uv0"],atol=2e-6)
        np.testing.assert_array_equal(layout,g["uv1"]);np.testing.assert_array_equal(color,g["color"])
        np.testing.assert_array_equal(faces,g["f"])
        idx=ref.decode(color[:,:2].astype(np.float32)/255)
        assert np.all(idx[faces]==idx[faces][:,:1]) and idx.max()<len(table)
        assert all(table[int(i)]["role"]==role for i in np.unique(idx))
        out.extend(pos[faces].reshape(-1,3));uv.extend(layout[faces].reshape(-1,2))
        ids.extend(idx[faces].reshape(-1));roles.extend([role]*len(faces))
    independent=trimesh.load(path,force="scene",process=False)
    assert sum(len(m.faces) for m in independent.geometry.values())==len(out)//3
    vv=np.array(out,dtype=np.float32)
    y=y_up(vv)
    np.testing.assert_allclose(independent.bounds,[y.min(0),y.max(0)],atol=1e-6)
    return vv,np.asarray(uv,dtype=np.float32),np.asarray(ids),roles

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path("tests/fixtures/l2_massing.json").read_text())
    groups,table,cells,records=compile_scene(cases)
    g2,t2,c2,r2=compile_scene(list(reversed(cases)))
    assert table==t2 and records==r2
    np.testing.assert_array_equal(cells,c2)
    for role in GROUPS:
        for k in groups[role]:np.testing.assert_array_equal(groups[role][k],g2[role][k])
    path=OUT/"tile.glb";export_glb(groups,path)
    v,uv,ids,roles=read_glb(path,groups,table)
    np.testing.assert_allclose((v.astype(np.float64)+ORIGIN)-ORIGIN,v,atol=1e-9)
    (OUT/"facade_cells.bin").write_bytes(cells.tobytes())
    (OUT/"parameters.json").write_text(json.dumps(table,indent=2))
    loaded_cells=np.frombuffer((OUT/"facade_cells.bin").read_bytes(),dtype=np.uint8)
    np.testing.assert_array_equal(loaded_cells,cells)
    maps={r["family"]:texture_maps(MATERIALS[r["family"]],256)["basecolor"] for r in table}
    original=ref.shader
    def compose(sample,r,maps,mode,dense=False):
        c,e=original(sample,r,maps,"bare",False)
        if r["role"]!="wall":return c,e
        role,variant,local=lookup(sample,r,loaded_cells)
        x,y=local.T
        opening=(role==1)|(role==2)
        glass=opening&(x>.08)&(x<.92)&(y>.08)&(y<.92)&(np.abs(x-.5)>.035)
        c[opening]=srgb_decode(np.array([.78,.77,.72]))
        c[glass]=srgb_decode(np.column_stack((.16+.27*y[glass],.24+.25*y[glass],.29+.26*y[glass])))
        c[role==3]=srgb_decode(np.array([.32,.22,.14]))
        if mode=="night":
            for i,name in enumerate(VARIANTS):e[glass&(variant==i)]=EMISSION[name]
        return c,e
    sheet=Image.new("RGB",(1280,672),(235,235,235))
    ImageDraw.Draw(sheet).text((8,8),"Reloaded GLB / canonical facade plan / CPU diffuse+emission",fill=(0,0,0))
    try:
        ref.shader=compose
        old=None
        for i,mode in enumerate(("day","night")):
            im,d=ref.render(v,uv,ids,table,maps,mode,center_override=(42,42,6),ppm_override=4.5)
            if old is None:old=d
            else:np.testing.assert_array_equal(d,old)
            im.save(OUT/(mode+".png"));sheet.paste(im,(i*640,32))
    finally:ref.shader=original
    sheet.save(OUT/"contact.png")
    manifest={"schema":"pcg.research.zero-tile.v1","synthetic_projected_origin":ORIGIN.tolist(),
              "axis":"GLB (x,z,-y), local metric Z-up sidecar; no CRS claim",
              "bounds_local":[v.min(0).tolist(),v.max(0).tolist()],
              "mesh":"tile.glb","parameter_table":"parameters.json","cell_table":"facade_cells.bin",
              "abi":{"TEXCOORD_0":"glTF V-down metric repeat coordinates","TEXCOORD_1":"raw unwrapped u,z meters, no V flip in file",
                     "COLOR_0":"normalized RGBA8, record index=Rbyte+256*Gbyte, B=0 A=255",
                     "cell_byte":"low two bits role 0 blank 1 window 2 balcony door, upper bits static emission variant; door exception in record"},
              "files":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ("tile.glb","parameters.json","facade_cells.bin")},
              "buildings":records,"limitations":["synthetic fixtures, one tile","balcony access intents retained but no balcony volumes emitted: not production-ready",
                "no native shader, dynamic room occupancy not integrated into canonical-plan lane","no foundation geometry, no complex concave pitched roof beyond existing explicit fallback",
                "CPU diffuse reference, not PBR or native SDK","generic GLB viewer will display ID colors, requires custom shader"]}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    report={"buildings":len(records),"triangles":len(v)//3,"indexed_vertices":sum(len(g["v"]) for g in groups.values()),
            "mesh_nodes":1,"role_primitives":3,"records":len(table),"cell_bytes":len(cells),
            "openings":sum(b["openings"] for b in records),"opening_triangles_added":0,
            "exact_layout_samples":4096*sum(b["facades"] for b in records),
            "roof_iou_min":min(b["roof_iou"] for b in records),
            "explicit_fallbacks":sum(b["fallback_reason"] is not None for b in records),
            "glb_bytes":path.stat().st_size,"glb_raw_attributes_roundtrip":True,"independent_trimesh_geometry_roundtrip":True,
            "input_order_invariant":True,"same_day_night_depth":True,"unique_image_textures":0,"native_sdk":"NOT_RUN",
            "limitations":manifest["limitations"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("ZERO_TILE_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1000,700));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=70,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"ZERO_TILE_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main()
