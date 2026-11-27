"""Independent disk-bundle verification before a native loader consumes it.
No compiler imports or reconstruction of expected generator objects.
Integrity checks are not a signature or substitute for untrusted-asset sandboxing.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

def need(ok,msg):
    if not ok:raise ValueError(msg)

def locate(root,base,name):
    p=(base/name).resolve(strict=True)
    need(p.is_relative_to(root) and p.is_file(),"bundle path outside root or not a file: "+str(name))
    return p

def read(root,base,name,digest=None,size=None):
    p=locate(root,base,name);b=p.read_bytes()
    if digest is not None:need(hashlib.sha256(b).hexdigest()==digest,"SHA256 mismatch: "+str(p))
    if size is not None:need(len(b)==size,"size mismatch: "+str(p))
    return p,b

def glb(data,records):
    need(len(data)>=28 and struct.unpack_from("<4sII",data)==(b"glTF",2,len(data)),"bad GLB header")
    jn,jt=struct.unpack_from("<I4s",data,12);need(jt==b"JSON","missing GLB JSON")
    tree=json.loads(data[20:20+jn])
    bn,bt=struct.unpack_from("<I4s",data,20+jn);blob=data[28+jn:]
    need(bt==b"BIN\x00" and len(blob)==bn,"bad GLB BIN chunk")
    need(len(tree["nodes"])==len(tree["meshes"])==1 and not tree.get("images"),"mesh/texture contract")
    def accessor(i,kind,types):
        a=tree["accessors"][i];v=tree["bufferViews"][a["bufferView"]]
        widths={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4};sizes={5121:1,5123:2,5125:4,5126:4}
        need(a["type"]==kind and a["componentType"] in types,"accessor type mismatch")
        width=widths[kind]*sizes[a["componentType"]];start=v.get("byteOffset",0)+a.get("byteOffset",0)
        count=a["count"];need(count>0 and v.get("buffer",0)==0 and v.get("byteStride",width)==width,"accessor layout mismatch")
        end=start+count*width
        need(0<=start<=end<=len(blob) and a.get("byteOffset",0)+count*width<=v["byteLength"],"accessor range")
        return a,memoryview(blob)[start:end]
    triangles=0;seen=set()
    for p in tree["meshes"][0]["primitives"]:
        role=p["extras"]["pcg_role"];need(role not in seen and role in ("wall","roof","base"),"duplicate/unknown role");seen.add(role)
        need(p.get("mode",4)==4,"not triangles")
        attrs=p["attributes"]
        pos,pb=accessor(attrs["POSITION"],"VEC3",(5126,))
        n,_=accessor(attrs["NORMAL"],"VEC3",(5126,))
        u,_=accessor(attrs["TEXCOORD_1"],"VEC2",(5126,))
        u0,_=accessor(attrs["TEXCOORD_0"],"VEC2",(5126,))
        c,cb=accessor(attrs["COLOR_0"],"VEC4",(5121,))
        need(pos["count"]==n["count"]==u["count"]==u0["count"]==c["count"] and c.get("normalized") is True,"vertex contract")
        need(all(math.isfinite(f[0]) for f in struct.iter_unpack("<f",pb)),"nonfinite vertices")
        ids=[b[0]+256*b[1] for b in struct.iter_unpack("<4B",cb)]
        need(max(ids)<records,"record ID out of bounds")
        a,ib=accessor(p["indices"],"SCALAR",(5121,5123,5125))
        fmt={5121:"<B",5123:"<H",5125:"<I"}[a["componentType"]]
        inds=[v[0] for v in struct.iter_unpack(fmt,ib)]
        need(len(inds)%3==0 and max(inds)<len(ids),"index range")
        need(all(ids[inds[j]]==ids[inds[j+1]]==ids[inds[j+2]] for j in range(0,len(inds),3)),"nonconstant triangle ID")
        triangles+=len(inds)//3
    need(seen=={"wall","roof","base"},"missing role")
    return triangles

def verify(index,root):
    root=Path(root).resolve(strict=True);index=Path(index).resolve(strict=True)
    need(index.is_relative_to(root),"index outside package root")
    d=json.loads(index.read_bytes());need(d["schema"]=="pcg.research.zero-tiles.output.v1","unsupported index")
    lp,lb=read(root,index.parent,d["library"],d["library_sha256"]);library=json.loads(lb)
    need(library["schema"]=="pcg.research.shared-pbr.v1","library version")
    layers=library["layers"];need([r["layer"] for r in layers]==list(range(len(layers))),"layer order")
    need(len({r["material"] for r in layers})==len(layers),"duplicate material")
    maps=0
    for layer in layers:
        need(set(layer["maps"])=={"basecolor","normal","orm"},"missing material channel")
        for m in layer["maps"].values():
            read(root,lp.parent,m["path"],m["sha256"]);maps+=1
    ids=set();keys=set();tri=0;recs=0;cells=0;mesh_bytes=0;param_bytes=0;total=0
    for t in d["tiles"]:
        p,b=read(root,index.parent,t["manifest"],t["sha256"]);m=json.loads(b)
        key=tuple(m["tile"]);need(key==tuple(t["tile"]) and key not in keys,"duplicate tile");keys.add(key)
        need(m["schema"]=="pcg.research.zero-tile.v2","tile version")
        origin=d["world_frame"]["origin"];size=d["tile_size_m"]
        anchor=[origin[0]+key[0]*size,origin[1]+key[1]*size,origin[2]]
        need(m["world_anchor_enh"]==anchor,"tile anchor mismatch")
        need(set(m["files"])=={"tile.glb","facade_params.bin","facade_cells.bin","parameters.json"},"tile file set")
        files={}
        for name,meta in m["files"].items():
            _,files[name]=read(root,p.parent,name,meta["sha256"],meta["bytes"]);total+=meta["bytes"]
        r=files["facade_params.bin"];c=files["facade_cells.bin"];tab=json.loads(files["parameters.json"])
        need(len(r)>=16,"short ABI")
        magic,v,n,stride=struct.unpack_from("<4sIII",r)
        need((magic,v,stride)==(b"PCGF",1,112) and 1<=n<=65536 and len(r)==16+n*112,"ABI version or length")
        need(n==len(tab)==m["records"]==t["records"] and len(c)==m["cells"]==t["cells"],"ABI count mismatch")
        for i in range(n):
            u=struct.unpack_from("<12I",r,16+i*112)
            need(u[0]==i and u[1]<len(layers) and u[2]&3<3,"invalid ABI ID/layer/role")
            need(tab[i]["id"]==i and layers[u[1]]["material"]==tab[i]["family"],"diagnostic table mismatch")
        actual=glb(files["tile.glb"],n)
        need(actual==m["triangles"]==t["triangles"],"triangle count mismatch")
        owners=m["owned_buildings"]
        need(len(owners)==t["buildings"] and sum(o["triangles"] for o in owners)==actual,"owner count mismatch")
        for o in owners:
            need(o["building_id"] not in ids,"duplicate building owner");ids.add(o["building_id"])
        tri+=actual;recs+=n;cells+=len(c);mesh_bytes+=len(files["tile.glb"]);param_bytes+=len(r)
    result={"tiles":len(keys),"buildings":len(ids),"triangles":tri,"records":recs,
            "cell_bytes":cells,"glb_bytes":mesh_bytes,"parameter_bytes":param_bytes}
    need(all(d["report"][k]==v for k,v in result.items()),"index aggregate mismatch")
    return dict(result,verified_tile_bytes=total,shared_map_hashes=maps)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("index",type=Path);ap.add_argument("--root",type=Path,required=True)
    a=ap.parse_args();print(json.dumps(verify(a.index,a.root),sort_keys=True))
