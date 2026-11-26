"""Research-only explicit-input multi-tile compiler and numeric ABI validation.
Not a production GIS adapter, native shader, runtime streamer or SDK build.
"""
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import argparse
import base64
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import tempfile
import numpy as np
from PIL import Image,ImageDraw
from pcg.footprint import Footprint
from pcg.materials import MATERIALS,texture_maps
from pcg.occupancy import EMISSION
from pcg.far_emission import srgb_decode
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref

SCHEMA="pcg.research.zero-tiles.input.v1"
MAGIC=b"PCGF"
HEADER=struct.Struct("<4sIII")
RECORD=struct.Struct("<12I16f")
LAYERS=tuple(sorted(MATERIALS))
ROLE={"wall":0,"roof":1,"base":2}
U_FIELDS=("index","material_layer","role_flags","cell_offset","floor_count","bay_count",
          "surface_seed_lo","surface_seed_hi","building_seed_lo","building_seed_hi","reserved0","reserved1")
F_FIELDS=("length","height","floor_h","first_center","spacing","window_width","window_height","sill",
          "door_height","balcony_threshold","door_u","door_z","door_w","door_h","repeat_m","wear")
assert RECORD.size==112 and HEADER.size==16

def dump(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def number(x):
    if type(x) not in (int,float) or not math.isfinite(x):raise ValueError("nonfinite or nonnumeric input")
    return float(x)
def vec3(x):
    if not isinstance(x,list) or len(x)!=3:raise ValueError("expected array of three numbers")
    return [number(a) for a in x]
def seed64(key):
    return struct.unpack("<II",hashlib.sha256(key.encode()).digest()[:8])

def validate(data):
    if not isinstance(data,dict) or set(data)!={"schema","world_frame","tile_size_m","buildings"} or data["schema"]!=SCHEMA:
        raise ValueError("unknown input schema or fields")
    w=data["world_frame"]
    if not isinstance(w,dict) or set(w)!={"horizontal_crs","vertical_datum","origin","units","axis"} or w["units"]!="m" or w["axis"]!="ENH":
        raise ValueError("expected explicit projected metric ENH frame")
    if not all(isinstance(w[k],str) and w[k].strip() for k in ("horizontal_crs","vertical_datum")):
        raise ValueError("missing frame provenance")
    origin=vec3(w["origin"]);size=number(data["tile_size_m"])
    if not 16<=size<=4096:raise ValueError("unsupported tile size")
    if not isinstance(data["buildings"],list) or not data["buildings"]:raise ValueError("empty buildings")
    ids=set();out=[]
    required={"id","outer","holes","anchor","roof","rise_m","eave_m","floors","profile","party_edges","source_ref"}
    for b in data["buildings"]:
        if not isinstance(b,dict) or set(b)!=required:raise ValueError("invalid building fields")
        key=b["id"]
        if not isinstance(key,str) or not key.strip() or key in ids:raise ValueError("empty or duplicate building ID")
        ids.add(key)
        if not isinstance(b["source_ref"],str) or not b["source_ref"].strip():raise ValueError("missing source reference")
        f=Footprint(key,b["outer"],b["holes"])
        a=vec3(b["anchor"]);h=number(b["eave_m"]);rise=number(b["rise_m"])
        if h<=0 or type(b["floors"]) is not int or not 1<=b["floors"]<=200 or b["profile"] not in tile.PROFILES:
            raise ValueError("invalid building profile or height")
        if b["roof"] not in ("flat","hip","gable") or rise<0 or (b["roof"]=="flat")!=(rise==0):
            raise ValueError("invalid roof request")
        rings=(f.exterior,*f.holes)
        if not isinstance(b["party_edges"],list):raise ValueError("invalid party edge list")
        edges=[]
        for e in b["party_edges"]:
            if not isinstance(e,list) or len(e)!=2 or any(type(i) is not int for i in e):
                raise ValueError("invalid party edge")
            r,i=e
            if not (0<=r<len(rings) and 0<=i<len(rings[r])):raise ValueError("party edge outside canonical ring")
            edges.append(e)
        c=f.polygon().centroid
        owner=(math.floor((a[0]+c.x)/size),math.floor((a[1]+c.y)/size))
        local=[a[0]-owner[0]*size,a[1]-owner[1]*size,a[2]]
        # Keep authoritative footprint and source lineage independent of render coordinates.
        out.append((owner,{"id":key,"building_id":key,"outer":f.exterior,"holes":f.holes,
            "roof":b["roof"],"rise":rise,"eave_m":h,"floors":b["floors"],"profile":b["profile"],
            "party_edges":edges,"offset":local,"source_ref":b["source_ref"]}))
    return np.asarray(origin,dtype=np.float64),size,sorted(out,key=lambda p:p[1]["id"])

# Cross-precision comparison is a geometric contract, not a fuzzy role test.
# Reject any record whose corresponding opening boundary moves by >0.1 mm.
# This budget is for the SDK-free FP64 -> FP32 parameter conversion only.
MAX_ABI_EDGE_DRIFT_M=1e-4

def opening_rectangles(r,cells):
    """Independent explicit rectangles indexed by stable facade-local slot."""
    n=r["floors"]*r["bays"];a=r["cell_start"]
    assert 0<=a<=a+n<=len(cells)
    rect={}
    for j in np.flatnonzero(cells[a:a+n]):
        j=int(j);code=int(cells[a+j]);kind=code&3
        assert kind in (1,2) and code>>2<3
        floor,bay=divmod(j,r["bays"])
        center=r["first_center"]+bay*r["spacing"]
        bottom=floor*r["floor_h"]+(r["balcony_threshold"] if kind==2 else r["sill"])
        height=r["door_height"] if kind==2 else r["window_height"]
        width=r["window_width"]
        rect[j]=(code,(center-width/2,bottom,center+width/2,bottom+height))
    if r["door"] is not None:
        x,y,w,h=r["door"];rect[n]=(3,(x,y,x+w,y+h))
    return rect

def rectangle_oracle(sample,rect):
    """Explicit rectangle membership, independent of grid index arithmetic."""
    slot=np.full(len(sample),-1,dtype=np.int64)
    role=np.zeros(len(sample),np.uint8);var=np.zeros_like(role)
    local=np.zeros((len(sample),2),np.float64)
    u,z=sample.T
    for j,(code,(x0,y0,x1,y1)) in rect.items():
        assert x1>x0 and y1>y0
        m=(u>=x0)&(u<x1)&(z>=y0)&(z<y1)
        assert np.all(slot[m]==-1),"overlapping explicit opening rectangles"
        slot[m]=j;role[m]=code&3;var[m]=code>>2
        local[m]=np.column_stack(((u[m]-x0)/(x1-x0),(z[m]-y0)/(y1-y0)))
    return slot,role,var,local

def new_float32_audit():
    return {"boundary_samples":0,"max_edge_drift_m":0.,"edge_budget_m":MAX_ABI_EDGE_DRIFT_M,
            "boundary_examples":[]}

def audit_float32_lookup(sample,r,q,cells,audit):
    a=opening_rectangles(r,cells);b=opening_rectangles(q,cells)
    assert a.keys()==b.keys(),"opening identity changed during ABI conversion"
    drift=0.
    for j,(code,box) in a.items():
        other,box32=b[j]
        assert code==other,"opening role/emission changed during ABI conversion"
        drift=max(drift,max(abs(x-y) for x,y in zip(box,box32)))
    assert drift<=MAX_ABI_EDGE_DRIFT_M,(r["facade_id"],"ABI opening edge drift",drift)
    audit["max_edge_drift_m"]=max(audit["max_edge_drift_m"],drift)
    ai,er,ev,el=rectangle_oracle(sample,a)
    bi,fr,fv,fl=rectangle_oracle(sample,b)
    ar,av,al=tile.lookup(sample,r,cells);br,bv,bl=tile.lookup(sample,q,cells)
    # No ambiguous pixels are discarded from either exact classification test.
    np.testing.assert_array_equal(ar,er);np.testing.assert_array_equal(av,ev)
    np.testing.assert_array_equal(br,fr);np.testing.assert_array_equal(bv,fv)
    np.testing.assert_allclose(al,el,atol=1e-10,rtol=0)
    np.testing.assert_allclose(bl,fl,atol=1e-10,rtol=0)
    same=ai==bi
    np.testing.assert_array_equal(ar[same],br[same])
    np.testing.assert_array_equal(av[same],bv[same])
    np.testing.assert_allclose(al[same],bl[same],atol=1e-5)
    # A point can cross only the displaced edge of its corresponding opening.
    # A switch between two opening identities is never accepted.
    for k in np.flatnonzero(~same):
        k=int(k);assert (ai[k]<0)!=(bi[k]<0),"opening identity swap"
        j=int(max(ai[k],bi[k]));u,z=sample[k]
        def inside(box):
            x0,y0,x1,y1=box
            return x0<=u<x1 and y0<=z<y1
        assert inside(a[j][1]) != inside(b[j][1]),"not in exact boundary symmetric difference"
        assert (ar[k]!=br[k]) or (av[k]!=bv[k])
        audit["boundary_samples"]+=1
        if len(audit["boundary_examples"])<16:
            audit["boundary_examples"].append({"facade":r["facade_id"],"sample_index":k,
                "uv_m":sample[k].tolist(),"slot":j,"canonical_role":int(ar[k]),
                "decoded_role":int(br[k]),"canonical_rect_m":list(a[j][1]),
                "decoded_rect_m":list(b[j][1])})

def regression_float32_boundary():
    """Minimized version of the failed stress seed, not a relaxed random test."""
    fid="stress:00880:facade:0:0"
    r={"facade_id":fid,"floors":2,"bays":4,"cell_start":0,"door":None,
       "length":12.,"height":6.,"floor_h":3.,"first_center":1.8000000000000005,
       "spacing":2.8,"window_width":1.4,"window_height":1.5,"sill":.85,
       "door_height":2.2,"balcony_threshold":.05+.12}
    q=dict(r);q.update({k:float(np.float32(r[k])) for k in F_FIELDS[:10]})
    cells=np.zeros(8,np.uint8);cells[6]=2|(1<<2)
    point=np.random.default_rng(seed64(fid)[0]^0xc1c5).random((1024,2))[216]*[12.,6.]
    sample=np.array([point,point+[-1e-5,0],point+[1e-5,0]])
    np.testing.assert_array_equal(tile.lookup(sample,r,cells)[0],[2,2,0])
    np.testing.assert_array_equal(tile.lookup(sample,q,cells)[0],[0,2,0])
    audit=new_float32_audit();audit_float32_lookup(sample,r,q,cells,audit)
    assert audit["boundary_samples"]==1
    try:
        audit_float32_lookup(sample,r,dict(q,first_center=q["first_center"]+.01),cells,new_float32_audit())
    except AssertionError:pass
    else:raise AssertionError("out-of-budget edge displacement was accepted")
    return audit

def pack_table(table,cells,path,audit=None):
    if audit is None:audit=new_float32_audit()
    if not 1<=len(table)<=65536:raise ValueError("surface record limit; split spatial tile")
    buf=bytearray(HEADER.pack(MAGIC,1,len(table),RECORD.size))
    for i,r in enumerate(table):
        assert i==r["id"]
        wall=r["role"]=="wall";door=r.get("door")
        flags=ROLE[r["role"]]|(4 if r.get("party") else 0)|(8 if r.get("courtyard") else 0)|(16 if door else 0)
        sl,sh=seed64(r.get("facade_id",r["surface"]));bl,bh=seed64(r["building"])
        assert sl==r["seed"]
        u=(i,LAYERS.index(r["family"]),flags,r.get("cell_start",0) if wall else 0,
           r["floors"] if wall else 0,r.get("bays",0) if wall else 0,sl,sh,bl,bh,0,0)
        d=door if door else (0.,0.,0.,0.)
        vals=tuple(r.get(k,0.) for k in F_FIELDS[:10])+tuple(d)+(MATERIALS[r["family"]].repeat_m,.25+.6*(bl%997)/996)
        assert len(vals)==16 and all(math.isfinite(x) for x in vals)
        buf.extend(RECORD.pack(*u,*vals))
    path.write_bytes(buf)
    assert len(buf)==16+112*len(table)
    # Audit float32-decoded numeric lookup against JSON canonical records.
    blob=path.read_bytes();magic,version,count,stride=HEADER.unpack_from(blob)
    assert (magic,version,count,stride)==(MAGIC,1,len(table),112)
    checked=0;seen=np.zeros(len(cells),dtype=np.uint8)
    for i,r in enumerate(table):
        item=RECORD.unpack_from(blob,16+i*112)
        u=dict(zip(U_FIELDS,item[:12]));f=dict(zip(F_FIELDS,item[12:]))
        assert u["index"]==i and LAYERS[u["material_layer"]]==r["family"]
        assert u["surface_seed_lo"]==r["seed"]
        assert (u["role_flags"]&3)==ROLE[r["role"]]
        np.testing.assert_allclose(f["repeat_m"],MATERIALS[r["family"]].repeat_m,atol=1e-7)
        if r["role"]!="wall":continue
        q=dict(r)
        q.update({k:f[k] for k in F_FIELDS[:10]})
        q.update(bays=u["bay_count"],floors=u["floor_count"],cell_start=u["cell_offset"],
                 door=[f[k] for k in ("door_u","door_z","door_w","door_h")] if u["role_flags"]&16 else None)
        n=q["bays"]*q["floors"];a=q["cell_start"]
        assert 0<=a<=a+n<=len(cells)
        seen[a:a+n]+=1
        rng=np.random.default_rng(r["seed"]^0xc1c5)
        sample=rng.random((1024,2))*[r["length"],r["height"]]
        audit_float32_lookup(sample,r,q,cells,audit)
        checked+=len(sample)
    assert np.all(seen==1)
    return checked

def material_library(root,size):
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,name in enumerate(LAYERS):
        recipe=MATERIALS[name];maps=texture_maps(recipe,size)
        path=root/f"{i:02d}_{name}";path.mkdir(exist_ok=True)
        files={}
        for k in ("basecolor","normal","orm"):
            p=path/(k+".png")
            data=np.uint8(np.round(np.clip(maps[k],0,1)*255))
            Image.fromarray(data).save(p)
            np.testing.assert_array_equal(np.asarray(Image.open(p)),data)
            files[k]={"path":str(p.relative_to(root)),"sha256":sha(p)}
        rows.append({"layer":i,"material":name,"recipe":asdict(recipe),"recipe_digest":recipe.digest(),"maps":files})
    manifest={"schema":"pcg.research.shared-pbr.v1","resolution":size,"layers":rows,
        "channels":{"basecolor":"sRGB","normal":"linear OpenGL +Y","orm":"linear AO/roughness/metalness"},
        "limitations":"Original synthetic recipe PNGs; no native texture arrays, compression, mip filtering or calibrated photographic quality."}
    p=root/"manifest.json";dump(p,manifest);return p

def build_dataset(data,root,library,decoder=None,audit_samples=1024,preview=False):
    origin,size,items=validate(data);root.mkdir(parents=True,exist_ok=True)
    buckets=defaultdict(list)
    for owner,c in items:buckets[owner].append(c)
    seen=set();world_geometry={};semantic={};manifests=[];decoded_samples=0;cpp=0;largest=None
    precision_audit=new_float32_audit()
    source={c["id"]:c["source_ref"] for _,c in items}
    for owner,cases in sorted(buckets.items()):
        key=f"{owner[0]}_{owner[1]}";out=root/key;out.mkdir(exist_ok=True)
        groups,table,cells,records=tile.compile_scene(cases,audit_samples=audit_samples)
        for b in records:
            bid=b["building_id"];assert bid not in seen;seen.add(bid);b["source_ref"]=source[bid]
        mesh=out/"tile.glb";tile.export_glb(groups,mesh)
        v,uv,ids,roles=tile.read_glb(mesh,groups,table)
        (out/"facade_cells.bin").write_bytes(cells.tobytes())
        dump(out/"parameters.json",table)
        decoded_samples+=pack_table(table,cells,out/"facade_params.bin",precision_audit)
        if decoder:
            run=subprocess.run([str(decoder),str(out/"facade_params.bin"),str(out/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True)
            result=json.loads(run.stdout)
            assert result["records"]==len(table) and result["cells"]==len(cells)
            dump(out/"cpp_abi_check.json",result);cpp+=1
        shift=np.array([owner[0]*size,owner[1]*size,0.],dtype=np.float64)
        anchor=origin+shift
        for r in table:
            if r["role"]!="wall":continue
            a=r["cell_start"];n=r["floors"]*r["bays"]
            fid=r["facade_id"];assert fid not in semantic
            semantic[fid]=(r["seed"],r["family"],r["door"],cells[a:a+n].tobytes())
        for b in records:
            bid=b["building_id"]
            rid={r["id"] for r in table if r["building"]==bid}
            mask=np.isin(ids,list(rid));assert mask.sum()==3*b["triangles"]
            world_geometry[bid]=v[mask].astype(np.float64)+shift
        local_bounds=[v.min(0).astype(float).tolist(),v.max(0).astype(float).tolist()]
        world_bounds=[(np.asarray(p)+anchor).tolist() for p in local_bounds]
        files={name:{"bytes":(out/name).stat().st_size,"sha256":sha(out/name)} for name in ("tile.glb","facade_params.bin","facade_cells.bin","parameters.json")}
        m={"schema":"pcg.research.zero-tile.v2","tile":list(owner),"world_anchor_enh":anchor.tolist(),
            "bounds_local_z_up":local_bounds,"bounds_world_enh":world_bounds,"owned_buildings":records,
            "triangles":len(v)//3,"records":len(table),"cells":len(cells),"files":files}
        dump(out/"manifest.json",m)
        manifests.append({"tile":list(owner),"manifest":key+"/manifest.json","sha256":sha(out/"manifest.json"),
            "buildings":len(records),"triangles":len(v)//3,"records":len(table),"cells":len(cells),
            "glb_bytes":files["tile.glb"]["bytes"],"param_bytes":files["facade_params.bin"]["bytes"]})
        if largest is None or len(records)>largest[0]:largest=(len(records),v,uv,ids,table,cells)
    assert seen==set(source)
    normalized=dict(data,buildings=sorted(data["buildings"],key=lambda b:b["id"]))
    report={"tiles":len(manifests),"buildings":len(seen),"triangles":sum(m["triangles"] for m in manifests),
        "records":sum(m["records"] for m in manifests),"cell_bytes":sum(m["cells"] for m in manifests),
        "glb_bytes":sum(m["glb_bytes"] for m in manifests),"parameter_bytes":sum(m["param_bytes"] for m in manifests),
        "cpp_abi_tiles":cpp,"float32_lookup_samples":decoded_samples,"opening_triangles_added":0,
        "float32_boundary_audit":precision_audit,
        "max_records_per_tile":max(m["records"] for m in manifests),"unique_image_textures_per_building":0}
    index={"schema":"pcg.research.zero-tiles.output.v1","input_sha256":hashlib.sha256(canonical(normalized)).hexdigest(),
        "world_frame":data["world_frame"],"tile_size_m":size,"library":os.path.relpath(library,root),
        "library_sha256":sha(library),"abi":{"header":"<4sIII PCGF/version/count/stride",
            "record":"<12I16f little-endian","stride":112,"u32":U_FIELDS,"float32":F_FIELDS,
            "role_flags":"bits0-1 wall0/roof1/base2; bit2 party; bit3 courtyard; bit4 door",
            "cells":"uint8 low2 role, high bits static emission variant; per-facade offset + floor*bays+bay",
            "COLOR_0":"normalized RGBA8 index R+256G","TEXCOORD_1":"unwrapped metric facade u,z; raw GLB accessor"},
        "tiles":manifests,"report":report,"limitations":["research only","no native SDK or runtime streaming",
            "no native GPU ABI upload","no foundation or balcony volumes","static canonical occupancy",
            "explicit fallback roofs retained","no real-data validation"]}
    dump(root/"index.json",index)
    if preview:
        _,v,uv,ids,table,cells=largest
        maps={name:np.asarray(Image.open(library.parent/f"{LAYERS.index(name):02d}_{name}/basecolor.png"),dtype=float)/255 for name in {r["family"] for r in table}}
        old=ref.shader
        def compose(sample,r,maps,mode,dense=False):
            c,e=old(sample,r,maps,"bare",False)
            if r["role"]!="wall":return c,e
            role,var,local=tile.lookup(sample,r,cells);x,y=local.T
            op=(role==1)|(role==2)
            glass=op&(x>.08)&(x<.92)&(y>.08)&(y<.92)&(np.abs(x-.5)>.035)
            c[op]=srgb_decode(np.array([.78,.77,.72]))
            c[glass]=srgb_decode(np.column_stack((.16+.27*y[glass],.24+.25*y[glass],.29+.26*y[glass])))
            c[role==3]=srgb_decode(np.array([.32,.22,.14]))
            if mode=="night":
                for i,name in enumerate(tile.VARIANTS):e[glass&(var==i)]=EMISSION[name]
            return c,e
        lo=v.min(0);hi=v.max(0);center=(lo+hi)/2;ppm=560/max(1,np.linalg.norm(hi-lo))
        sheet=Image.new("RGB",(1280,672),(230,230,230))
        ImageDraw.Draw(sheet).text((8,8),"Largest actual reloaded tile / shared material source maps / CPU diagnostic",fill=(0,0,0))
        try:
            ref.shader=compose;depth=None
            for i,mode in enumerate(("day","night")):
                im,d=ref.render(v,uv,ids,table,maps,mode,center_override=center,ppm_override=ppm)
                if depth is None:depth=d
                else:np.testing.assert_array_equal(depth,d)
                sheet.paste(im,(i*640,32));im.save(root/(mode+".png"))
        finally:ref.shader=old
        sheet.save(root/"contact.png")
        thumb=sheet.copy();thumb.thumbnail((1000,700));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=65,optimize=True)
        payload=base64.b64encode(buf.getvalue()).decode()
        for i in range(0,len(payload),4000):print(f"MULTITILE_PREVIEW_{i//4000:03d} "+payload[i:i+4000],flush=True)
    return report,world_geometry,semantic

def fixture():
    cases=json.loads(Path("tests/fixtures/l2_massing.json").read_text())
    buildings=[]
    for i in range(96):
        c=cases[i%len(cases)]
        floors=2+i%3
        buildings.append({"id":f"synthetic:{i:04d}","outer":c["outer"],"holes":c.get("holes",[]),
            "anchor":[(i%12)*42.-100.,(i//12)*42.-80.,(i%5)*.25],
            "roof":c["roof"],"rise_m":c.get("rise",0.),"eave_m":floors*3.,"floors":floors,
            "profile":("house","apartments","office")[i%3],"party_edges":[],
            "source_ref":"original synthetic l2_massing fixture:"+c["id"]})
    return {"schema":SCHEMA,"world_frame":{"horizontal_crs":"LOCAL_SYNTHETIC",
        "vertical_datum":"SYNTHETIC_ZERO","origin":[700000.,4800000.,125.],"units":"m","axis":"ENH"},
        "tile_size_m":128,"buildings":buildings}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path)
    ap.add_argument("--out",type=Path,default=Path("artifacts/zero-multitile"))
    ap.add_argument("--texture-size",type=int,choices=(512,1024),default=512)
    ap.add_argument("--decoder",type=Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test==(args.input is not None):ap.error("choose exactly one --self-test or --input")
    if args.self_test and not args.decoder:ap.error("self-test requires compiled independent C++ decoder")
    decoder=args.decoder.resolve() if args.decoder else None
    args.out.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        data=fixture()
        # Fail closed on invalid inputs before compiling.
        for bad in (dict(data,buildings=data["buildings"]+[data["buildings"][0]]),
                    dict(data,world_frame=dict(data["world_frame"],origin=[0,0,float("nan")])),
                    dict(data,buildings=[dict(data["buildings"][0],roof="unknown")]),
                    dict(data,buildings=[dict(data["buildings"][0],party_edges=[[99,0]])])):
            try:validate(bad)
            except ValueError:pass
            else:raise AssertionError("invalid input accepted")
        lib=material_library(args.out/"library",args.texture_size)
        dump(args.out/"input_128.json",data)
        a,ga,sa=build_dataset(data,args.out/"tiles128",lib,decoder,preview=True)
        d2=dict(data,tile_size_m=256,buildings=list(reversed(data["buildings"])))
        dump(args.out/"input_256.json",d2)
        b,gb,sb=build_dataset(d2,args.out/"tiles256",lib,decoder)
        assert sa==sb and ga.keys()==gb.keys()
        max_delta=0.
        for k in ga:
            np.testing.assert_allclose(ga[k],gb[k],atol=5e-5,rtol=0)
            max_delta=max(max_delta,float(np.abs(ga[k]-gb[k]).max()))
        assert a["triangles"]==b["triangles"] and a["buildings"]==b["buildings"]==96
        # Decoder rejection tests: damaged header, truncation, invalid local record index.
        original=next((args.out/"tiles128").glob("*/facade_params.bin"))
        original_cells=original.with_name("facade_cells.bin")
        raw=original.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.bin"
            variants=[b"BAD!"+raw[4:],raw[:-1],raw[:16]+struct.pack("<I",12345)+raw[20:]]
            for bad in variants:
                p.write_bytes(bad)
                proc=subprocess.run([str(decoder),str(p),str(original_cells),str(len(LAYERS))],capture_output=True)
                assert proc.returncode!=0,"corrupt binary accepted"
        result={"tiles128":a,"tiles256":b,"retile_max_position_delta_m":max_delta,
            "semantic_state_exact_across_retile":True,"shared_material_layers":len(LAYERS),
            "source_map_resolution":args.texture_size,"invalid_inputs_rejected":4,"corrupt_binary_rejected":3,
            "scope":"96 actual synthetic buildings per tile-size lane; GLB raw and independent geometry readback, C++ numeric ABI; no SDK/GPU"}
        dump(args.out/"report.json",result);print("MULTITILE_REPORT "+json.dumps(result),flush=True)
    else:
        data=json.loads(args.input.read_text());validate(data)
        lib=material_library(args.out/"library",args.texture_size)
        result,_,_=build_dataset(data,args.out/"dataset",lib,decoder)
        dump(args.out/"report.json",result);print(json.dumps(result))

if __name__=="__main__":main()
