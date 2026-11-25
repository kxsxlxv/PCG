"""Research-only CPU reference: merged flat-shell facades, byte IDs, opening shading.
Reuses pcg.geometry, material maps and linear color conversion. Not UUSL/PBR/GPU.
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from pcg.contracts import Rectangle
from pcg.geometry import build_shell
from pcg.materials import MATERIALS, texture_maps
from pcg.far_emission import srgb_encode, srgb_decode

OUT=Path("artifacts/zero-geometry")
OUT.mkdir(parents=True,exist_ok=True)
WALLS=("plaster_cream","red_brick","concrete_panel","plaster_white")
ROOFS=("ceramic_roof","bitumen_roof","metal_roof_red")

def seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],"little")

def pack(ids):
    x=np.asarray(ids)
    if x.dtype.kind not in "iu" or np.any(x<0) or np.any(x>65535):
        raise ValueError("index outside uint16")
    x=x.astype(np.uint32)
    return np.stack((x&255,x>>8),-1).astype(np.uint8)

def decode(x):
    b=np.rint(np.asarray(x)*255).astype(np.uint32)
    return b[...,0]+256*b[...,1]

def build():
    vertices=[];uv=[];encoded=[];roles=[];table=[]
    for i in range(12):
        key=f"fixture:{i}"; floors=2+i%4
        w=11+2*(i%3); d=8+i%2; h=3*floors
        roof="gable" if floors==2 else "flat"
        b=Rectangle(key,w,d,h,roof,2.5 if roof=="gable" else 0.)
        shell=build_shell(b)
        offset=np.array([(i%4)*22,(i//4)*22,0.])
        v=np.asarray(shell.vertices)
        for s in shell.surfaces:
            p=v[list(s.vertices)]
            rid=len(table)
            edge=p[1]-p[0]; length=np.linalg.norm(edge)
            if s.role=="wall":
                coords=np.column_stack(((p-p[0])@(edge/length),p[:,2]))
                # Gable above eave is intentionally blank.
                bays=max(2,round(length/3))
                family=WALLS[i%len(WALLS)]
            else:
                coords=p[:,:2];bays=0;family=ROOFS[i%len(ROOFS)] if s.role=="roof" else "stone_plinth"
            record={"id":rid,"building":key,"surface":s.id,"seed":seed(s.id),"family":family,
                    "role":s.role,"floors":floors,"height":h,"bays":bays,"length":float(length),
                    "entry":s.id.endswith(":south")}
            table.append(record)
            for j in range(1,len(p)-1):
                ix=[0,j,j+1]
                vertices.extend(p[ix]+offset)
                uv.extend(coords[ix])
                encoded.extend([pack([rid])[0]]*3)
                roles.append(s.role)
    return np.array(vertices,dtype=np.float32),np.array(uv,dtype=np.float32),np.array(encoded,dtype=np.uint8),roles,table

def shader(uv,r,maps,mode,dense=False):
    u,z=uv.T
    a=maps[r["family"]]
    scale=MATERIALS[r["family"]].repeat_m
    ix=np.floor((u/scale%1)*a.shape[1]).astype(int)%a.shape[1]
    iy=np.floor((-z/scale%1)*a.shape[0]).astype(int)%a.shape[0]
    color=srgb_decode(a[iy,ix].copy())
    emission=np.zeros_like(color)
    if r["role"]!="wall":return color,emission
    tint=.90+.20*((r["seed"]%997)/996)
    color*=tint
    if mode=="bare":return color,emission
    bays=r["bays"]*(2 if dense else 1)
    gx=u/r["length"]*bays; gy=z/3.
    bx=np.floor(gx).astype(np.int64);fl=np.floor(gy).astype(np.int64)
    x=gx%1;y=gy%1
    valid=(bx>=0)&(bx<bays)&(fl>=0)&(fl<r["floors"])
    win=valid&(x>.23)&(x<.77)&(y>.25)&(y<.82)
    door=valid&r["entry"]&(fl==0)&(bx==bays//2)&(x>.23)&(x<.77)&(y<.80)
    win &= ~door
    glass=win&(x>.27)&(x<.73)&(y>.29)&(y<.78)&(np.abs(x-.5)>.025)&(np.abs(y-.54)>.018)
    color[win]=srgb_decode(np.array([.79,.78,.73]))
    color[glass]=srgb_decode(np.column_stack((.20+.24*y[glass],.27+.24*y[glass],.30+.28*y[glass])))
    color[door]=srgb_decode(np.array([.27,.18,.12]))
    if mode=="night":
        # CPU hash reference, room groups of two bays. No per-opening meshes/lights.
        room=(bx//2).astype(np.uint64)
        hh=(np.uint64(r["seed"])+room*374761393+fl.astype(np.uint64)*668265263)&np.uint64(0xffffffff)
        hh=((hh^(hh>>13))*np.uint64(1274126177))&np.uint64(0xffffffff)
        hh=hh^(hh>>16)
        lit=glass&((hh%1000)<380)
        emission[lit]=np.array([.82,.52,.24])
        emission[lit&(hh%4==0)]=np.array([.66,.69,.57])
    return color,emission

def render(v,uv,ids,table,maps,mode,dense=False,close=False):
    size=640
    view=np.array([1.,-1.5,.85]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    center=np.array([6.,4.,4.]) if close else np.array([39.,27.,5.])
    ppm=34 if close else 6.1
    pts=v-center
    xy=np.column_stack((pts@right,-pts@up))*ppm+size/2
    z=pts@view
    img=np.full((size,size,3),[15,22,32] if mode=="night" else [224,229,231],dtype=np.uint8)
    depth=np.full((size,size),-np.inf)
    light=np.array([.4,-.6,.7]);light/=np.linalg.norm(light)
    for k in range(0,len(v),3):
        if close and table[int(ids[k])]["building"]!="fixture:0":continue
        p=v[k:k+3];n=np.cross(p[1]-p[0],p[2]-p[0]);n/=np.linalg.norm(n)
        if n@view<=1e-9:continue
        a,b,c=xy[k:k+3]
        lo=np.maximum(np.floor(np.min([a,b,c],axis=0)).astype(int),0)
        hi=np.minimum(np.ceil(np.max([a,b,c],axis=0)).astype(int)+1,size)
        x0,y0=lo;x1,y1=hi
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];xx=xx+.5;yy=yy+.5
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        wc=1-wa-wb
        dz=wa*z[k]+wb*z[k+1]+wc*z[k+2];dst=depth[y0:y1,x0:x1]
        mask=(wa>=-1e-8)&(wb>=-1e-8)&(wc>=-1e-8)&(dz>dst)
        dst[mask]=dz[mask]
        sample=wa[mask,None]*uv[k]+wb[mask,None]*uv[k+1]+wc[mask,None]*uv[k+2]
        color,emission=shader(sample,table[int(ids[k])],maps,mode,dense)
        shade=.68+.32*max(0,n@light)
        color=color*shade*(.055 if mode=="night" else 1.)+emission
        img[y0:y1,x0:x1][mask]=np.uint8(np.round(srgb_encode(color)*255))
    assert np.isfinite(depth).sum()>100
    return Image.fromarray(img),depth

def main():
    all_ids=np.arange(65536,dtype=np.uint32)
    np.testing.assert_array_equal(decode(pack(all_ids).astype(np.float32)/255),all_ids)
    for invalid in (-1,65536):
        try:pack([invalid])
        except ValueError:pass
        else:raise AssertionError("silent index overflow")
    v,uv,encoded,roles,table=build()
    np.savez_compressed(OUT/"merged_mesh.npz",positions=v,uv0=uv,id_bytes=encoded)
    (OUT/"parameters.json").write_text(json.dumps(table,indent=2))
    with np.load(OUT/"merged_mesh.npz") as f:
        v=f["positions"];uv=f["uv0"];ids=decode(f["id_bytes"].astype(np.float32)/255)
    assert np.isfinite(v).all() and np.isfinite(uv).all()
    assert np.all(ids.reshape(-1,3)==ids.reshape(-1,3)[:,:1])
    assert ids.max()<len(table)
    assert len({r["id"] for r in table})==len(table)
    assert np.all(np.linalg.norm(np.cross(v[1::3]-v[::3],v[2::3]-v[::3]),axis=1)>1e-8)
    maps={name:texture_maps(MATERIALS[name],256)["basecolor"] for name in {r["family"] for r in table}}
    sheet=Image.new("RGB",(640*3,640*2+32),(240,240,240))
    ImageDraw.Draw(sheet).text((8,8),"CPU diagnostic: exact same merged mesh / plain, window masks, procedural night",fill=(0,0,0))
    for row,close in enumerate((True,False)):
        reference=None
        for col,mode in enumerate(("bare","day","night")):
            im,dep=render(v,uv,ids,table,maps,mode,close=close)
            if reference is None:reference=dep
            else:np.testing.assert_array_equal(dep,reference)
            ImageDraw.Draw(im).text((8,8),mode,fill=(220,220,220) if mode=="night" else (0,0,0))
            im.save(OUT/f"{'close' if close else 'block'}_{mode}.png")
            sheet.paste(im,(col*640,row*640+32))
        im,dep=render(v,uv,ids,table,maps,"day",dense=True,close=close)
        np.testing.assert_array_equal(dep,reference)
        im.save(OUT/f"{'close' if close else 'block'}_dense.png")
    sheet.save(OUT/"contact.png")
    groups_by_material=len({(r["role"],r["family"]) for r in table})
    report={"buildings":12,"triangles":len(v)//3,"split_vertices":len(v),
            "surface_records":len(table),"separate_role_material_groups":groups_by_material,
            "shared_role_groups":len(set(roles)),"opening_triangles_added":0,
            "id_roundtrip_count":65536,"same_depth_across_material_and_density_lanes":True,
            "rectangular_flat_5floor_idealized_counts":{"whole_wall":8,"four_zones":32,"per_floor":40},
            "renderer":"CPU orthographic barycentric raster, albedo and linear emission only; no PBR or native SDK",
            "limitations":["synthetic rectangular shells","nearest sampling, no mip/AA","NPZ roundtrip only, no GLB/native attribute test","zone counts analytical, not emitted","no native GPU or draw call measurements"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("ZERO_GEOMETRY_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((960,800));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=65,optimize=True)
    s=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(s),4000):print(f"ZERO_GEOMETRY_PREVIEW_{i//4000:03d} "+s[i:i+4000],flush=True)

if __name__=="__main__":main()
