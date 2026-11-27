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

def render(v,uv,ids,table,maps,mode,dense=False,close=False,center_override=None,ppm_override=None,view_override=None,visibility=None,lighting=None,capture_ids=None,two_sided=None,resolution=(640,640),depth_only=False):
    width,height=map(int,resolution)
    if width<=0 or height<=0:raise ValueError("invalid resolution")
    if capture_ids is not None and capture_ids.shape!=(height,width):raise ValueError("capture_ids size mismatch")
    view=np.array([1.,-1.5,.85] if view_override is None else view_override,dtype=float);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    center=np.array([6.,4.,4.]) if close else np.array([39.,27.,5.])
    ppm=34 if close else 6.1
    if center_override is not None:center=np.asarray(center_override,dtype=float)
    if ppm_override is not None:ppm=float(ppm_override)
    pts=v-center
    xy=np.column_stack((pts@right,-pts@up))*(ppm*height/640.)+np.array([width,height])/2
    z=pts@view
    img=np.full((height,width,3),[15,22,32] if mode=="night" else [224,229,231],dtype=np.uint8)
    depth=np.full((height,width),-np.inf)
    light=np.array([.4,-.6,.7]);light/=np.linalg.norm(light)
    for k in range(0,len(v),3):
        if close and table[int(ids[k])]["building"]!="fixture:0":continue
        p=v[k:k+3];n=np.cross(p[1]-p[0],p[2]-p[0]);n/=np.linalg.norm(n)
        record=table[int(ids[k])]
        if n@view<=1e-9:
            if two_sided is None or not two_sided(record) or abs(n@view)<=1e-9:continue
            n=-n
        a,b,c=xy[k:k+3]
        lo=np.maximum(np.floor(np.min([a,b,c],axis=0)).astype(int),0)
        hi=np.minimum(np.ceil(np.max([a,b,c],axis=0)).astype(int)+1,[width,height])
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
        sample=wa[mask,None]*uv[k]+wb[mask,None]*uv[k+1]+wc[mask,None]*uv[k+2]
        if visibility is not None:
            keep=visibility(sample,record)
            assert keep.shape==(len(sample),) and keep.dtype==bool
            mask[mask]=keep;sample=sample[keep]
        dst[mask]=dz[mask]
        if capture_ids is not None:capture_ids[y0:y1,x0:x1][mask]=int(ids[k])
        if depth_only:continue
        color,emission=shader(sample,record,maps,mode,dense)
        shade=.68+.32*max(0,n@light)
        if lighting is not None:
            world=wa[mask,None]*p[0]+wb[mask,None]*p[1]+wc[mask,None]*p[2]
            shade=lighting(world,n,record)[:,None]
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
    material_variation(v,uv,ids,table,maps)
    night_and_spatial(v,uv,ids,table,maps)
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


def material_variation(v,uv,ids,table,maps):
    """Grouped same-mesh experiment. Shared procedural field, not a per-building image."""
    from pcg._material_patterns import noise
    global shader
    original=shader
    field=noise(256,12,np.random.default_rng(872))
    # All buildings use the same shared mask; stable offsets select distinct regions.
    def sample_field(u,z,scale,offset):
        x=(u/scale+offset[0])%1*256
        y=(z/scale+offset[1])%1*256
        i=np.floor(x).astype(int);j=np.floor(y).astype(int)
        fx=x-i;fy=y-j
        return ((1-fx)*(1-fy)*field[j,i]+fx*(1-fy)*field[j,(i+1)%256]
                +(1-fx)*fy*field[(j+1)%256,i]+fx*fy*field[(j+1)%256,(i+1)%256])
    def variation(sample,r,maps,mode,dense=False):
        if mode not in ("macro","weather","weather_night"):
            return original(sample,r,maps,mode,dense)
        daymode="night" if mode=="weather_night" else "day"
        c,e=original(sample,r,maps,daymode,dense)
        if r["role"]!="wall":return c,e
        u,z=sample.T
        gx=u/r["length"]*r["bays"]*(2 if dense else 1);gy=z/3
        bx=np.floor(gx).astype(int);fl=np.floor(gy).astype(int)
        x=gx%1;y=gy%1
        valid=(bx>=0)&(bx<r["bays"]*(2 if dense else 1))&(fl>=0)&(fl<r["floors"])
        win=valid&(x>.23)&(x<.77)&(y>.25)&(y<.82)
        door=valid&r["entry"]&(fl==0)&(bx==r["bays"]*(2 if dense else 1)//2)&(x>.23)&(x<.77)&(y<.80)
        wall=~(win|door)
        a=seed(r["building"]);b=r["seed"]
        offset=((b&65535)/65536,(b>>16)/65536)
        macro=sample_field(u,z,38.,offset)
        broad=sample_field(u,z,105.,((a&65535)/65536,(a>>16)/65536))
        c[wall]*=(.76+.35*macro[wall]+.15*broad[wall])[:,None]
        if mode in ("weather","weather_night"):
            age=.25+.6*((a%997)/996)
            ground=np.exp(-np.maximum(z,0)/.65)*(.4+.6*macro)
            # Short runoff below each sill, limited to its horizontal neighborhood.
            streak=(np.maximum(0,1-np.abs(x-.28)/.065)+np.maximum(0,1-np.abs(x-.72)/.065))
            sill=valid*streak*np.exp(-np.maximum(.25-y,0)*7)*(y<.25)
            eave=np.exp(-np.abs(r["height"]-z)/.45)*(.2+.8*macro)
            patch=np.clip((broad-.69)/.25,0,1)*.3
            dirt=np.clip(age*(.40*ground+.25*sill+.20*eave+patch),0,.65)
            c[wall]=c[wall]*(1-dirt[wall,None])+srgb_decode(np.array([.26,.23,.17]))*dirt[wall,None]
        return c,e
    # Reference renderer keys the night exposure off mode; use a wrapper for night.
    sheet=Image.new("RGB",(640*3,640*2+32),(240,240,240))
    ImageDraw.Draw(sheet).text((8,8),"Same geometry / baseline, macro breakup, semantic weathering / CPU diffuse",fill=(0,0,0))
    report={"shared_mask_count":1,"shared_mask_size":256,"lanes":[],"renderer":"CPU diffuse, nearest base sampling; no PBR or SDK"}
    try:
        shader=variation
        for row,close in enumerate((True,False)):
            reference=None
            for col,mode in enumerate(("day","macro","weather")):
                im,dep=render(v,uv,ids,table,maps,mode,close=close)
                if reference is None:reference=dep
                else:np.testing.assert_array_equal(dep,reference)
                ImageDraw.Draw(im).text((8,8),mode,fill=(0,0,0))
                im.save(OUT/f"variation_{'close' if close else 'block'}_{mode}.png")
                sheet.paste(im,(col*640,row*640+32))
                report["lanes"].append({"mode":mode,"view":"close" if close else "block","same_depth":True})
        # Explicit semantic isolation, deterministic output and unchanged night emission.
        checked=0
        for r in table:
            if r["role"]!="wall":continue
            u,z=np.meshgrid(np.linspace(.001,r["length"]-.001,120),np.linspace(.001,r["height"]-.001,100))
            samples=np.column_stack((u.ravel(),z.ravel()))
            c0,e0=original(samples,r,maps,"night")
            c1,e1=variation(samples,r,maps,"weather_night")
            c2,e2=variation(samples,r,maps,"weather_night")
            np.testing.assert_array_equal(c1,c2);np.testing.assert_array_equal(e1,e2)
            np.testing.assert_array_equal(e0,e1)
            lit=np.any(e0>0,axis=1)
            np.testing.assert_array_equal(c0[lit],c1[lit])
            assert np.isfinite(c1).all() and c1.min()>=0
            checked+=len(samples)
        report["semantic_samples_checked"]=checked
        report["night_emission_unchanged"]=True
        report["lit_glass_color_unchanged"]=True
        report["triangles"]=len(v)//3
        report["opening_triangles_added"]=0
    finally:
        shader=original
    sheet.save(OUT/"variation_contact.png")
    (OUT/"variation_report.json").write_text(json.dumps(report,indent=2))
    print("VARIATION_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((960,800));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=65,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"VARIATION_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)


def night_and_spatial(v,uv,ids,table,maps):
    """One grouped CI: room occupancy, exact box filter, and spatial metadata stress.
    CPU reference only, not runtime shader or native batching. Diagnostic images are
    never stored as per-building runtime textures.
    """
    import time
    from pcg.far_emission import linear_mips
    global shader
    original=shader
    hours=(17,21,1)
    profiles={
        "house":(.18,.68,.10),
        "apartments":(.22,.57,.12),
        "office":(.65,.16,.025),
    }
    def profile(r):
        i=int(r["building"].split(":")[-1])
        return "office" if i%5==4 else "house" if r["floors"]==2 else "apartments"
    def h32(a,b,c):
        # Pure integer hash reference, inputs are stable identity and cell indices.
        x=(np.uint64(a)+np.asarray(b,dtype=np.uint64)*374761393+
           np.asarray(c,dtype=np.uint64)*668265263)&np.uint64(0xffffffff)
        x=((x^(x>>13))*np.uint64(1274126177))&np.uint64(0xffffffff)
        return x^(x>>16)
    def radiance(r,bay,floor,hour,correlated=True):
        room=bay//2 if correlated else bay
        h=h32(r["seed"],room,floor)
        q=h32(r["seed"]^0xa7b193e7,bay,floor)
        prob=profiles[profile(r)][hours.index(hour)]
        on=(h%10000)<round(prob*10000)
        warm=(h%4)!=0
        col=np.where(np.asarray(warm)[...,None],np.array([.82,.52,.24]),np.array([.66,.69,.57]))
        gain=.35+.65*(q%1000)/999.
        return col*(on*gain)[...,None],on
    def composite(sample,r,maps,mode,dense=False):
        color,e=original(sample,r,maps,"day",dense)
        if r["role"]!="wall":return color,e
        u,z=sample.T
        bays=r["bays"]*(2 if dense else 1)
        gx=u/r["length"]*bays;gy=z/3.
        bx=np.floor(gx).astype(int);fl=np.floor(gy).astype(int)
        x=gx%1;y=gy%1
        valid=(bx>=0)&(bx<bays)&(fl>=0)&(fl<r["floors"])
        door=valid&r["entry"]&(fl==0)&(bx==bays//2)&(x>.23)&(x<.77)&(y<.80)
        glass=valid&~door&(x>.27)&(x<.73)&(y>.29)&(y<.78)&(np.abs(x-.5)>.025)&(np.abs(y-.54)>.018)
        if np.any(glass):
            e[glass]=radiance(r,bx[glass],fl[glass],current_hour,current_correlated)[0]
        return color,e
    # Actual triangle-buffer renders at fixed cameras; independent/correlated
    # and temporal profiles evaluated without changing geometry.
    sheet=Image.new("RGB",(640*3,640*2+32),(18,25,35))
    ImageDraw.Draw(sheet).text((8,8),"CPU reference / room-correlated lighting / same merged shell",fill=(225,225,225))
    try:
        shader=composite
        for row,close in enumerate((True,False)):
            reference=None
            for col,current_hour in enumerate(hours):
                current_correlated=True
                im,dep=render(v,uv,ids,table,maps,"night",close=close)
                if reference is None:reference=dep
                else:np.testing.assert_array_equal(dep,reference)
                ImageDraw.Draw(im).text((8,8),f"{current_hour:02d}:00",fill=(230,230,230))
                im.save(OUT/f"night_{'close' if close else 'block'}_{current_hour}.png")
                sheet.paste(im,(col*640,row*640+32))
        current_hour=21;current_correlated=False
        im,_=render(v,uv,ids,table,maps,"night")
        im.save(OUT/"night_independent.png")
    finally:shader=original
    sheet.save(OUT/"night_contact.png")

    # Exact rectangular glass-pane coverage on a facade-aligned pixel grid.
    # This is a ground-truth CPU filter, not an efficient proposed runtime loop.
    def panes(r,hour,correlated=True):
        width=r["length"]/r["bays"]
        result=[]
        for floor in range(r["floors"]):
            for bay in range(r["bays"]):
                if r["entry"] and floor==0 and bay==r["bays"]//2:continue
                rgb,on=radiance(r,bay,floor,hour,correlated)
                for a,b in ((.27,.475),(.525,.73)):
                    for c,d in ((.29,.522),(.558,.78)):
                        result.append(((bay+a)*width,(bay+b)*width,(floor+c)*3,(floor+d)*3,rgb))
        return result
    def filter_rects(rects,xe,ze):
        dx=np.diff(xe);dz=np.diff(ze)
        image=np.zeros((len(dz),len(dx),3),np.float64)
        point=np.zeros_like(image)
        xc=(xe[1:]+xe[:-1])/2;zc=(ze[1:]+ze[:-1])/2
        total=np.zeros(3)
        for a,b,c,d,rgb in rects:
            ox=np.maximum(0,np.minimum(xe[1:],b)-np.maximum(xe[:-1],a))/dx
            oz=np.maximum(0,np.minimum(ze[1:],d)-np.maximum(ze[:-1],c))/dz
            image+=oz[:,None,None]*ox[None,:,None]*rgb
            point+=((zc[:,None]>=c)&(zc[:,None]<d)&(xc[None,:]>=a)&(xc[None,:]<b))[...,None]*rgb
            total+=(b-a)*(d-c)*rgb
        area=dz[:,None]*dx[None,:]
        np.testing.assert_allclose((image*area[...,None]).sum((0,1)),total,rtol=1e-12,atol=1e-10)
        return image,point,total,(point*area[...,None]).sum((0,1))
    error=[];phase_error=[];checks=0;windows=0;lit_counts={str(h):0 for h in hours};pairs=0
    for r in table:
        if r["role"]!="wall":continue
        # Actual stability under an ephemeral record index change.
        rr=dict(r,id=r["id"]+2000)
        for f in range(r["floors"]):
            for b in range(r["bays"]):
                if r["entry"] and f==0 and b==r["bays"]//2:continue
                windows+=1
                for hour in hours:
                    a,on=radiance(r,b,f,hour)
                    a2,on2=radiance(rr,b,f,hour)
                    np.testing.assert_array_equal(a,a2);assert on==on2
                    lit_counts[str(hour)]+=int(on)
                if b%2==0 and b+1<r["bays"] and not (r["entry"] and f==0 and r["bays"]//2 in (b,b+1)):
                    assert radiance(r,b,f,21)[1]==radiance(r,b+1,f,21)[1]
                    pairs+=1
        rects=panes(r,21)
        xe=np.linspace(0,r["length"],65);ze=np.linspace(0,r["height"],65)
        a,_,total,_=filter_rects(rects,xe,ze)
        levels=linear_mips(a)
        np.testing.assert_allclose(levels[-1][0,0],total/(r["length"]*r["height"]),atol=1e-12)
        # Phase-shifted grid, padded to cover every pane, exact energy vs point sampling.
        for pitch in (.5,1.,2.,4.):
            point_totals=[]
            for phase in np.arange(8)/8:
                xe=np.arange(-pitch+phase*pitch,r["length"]+2*pitch,pitch)
                ze=np.arange(-pitch+phase*pitch,r["height"]+2*pitch,pitch)
                _,_,total,point_total=filter_rects(rects,xe,ze)
                checks+=1
                if total.sum()>1e-9:
                    error.append(float(abs(point_total.sum()-total.sum())/total.sum()))
                    point_totals.append(float(point_total.sum()/total.sum()))
            if point_totals:phase_error.append(float(np.std(point_totals)))
    report={"hours":list(hours),"illustrative_probabilities":profiles,"windows":windows,
            "lit_counts":lit_counts,"correlated_pairs_checked":pairs,
            "exact_box_filter_checks":checks,"linear_mip_mean_preserved":True,
            "point_sampling_mean_relative_energy_error":float(np.mean(error)),
            "point_sampling_p95_relative_energy_error":float(np.percentile(error,95)),
            "point_sampling_mean_phase_std":float(np.mean(phase_error)),
            "record_index_independent":True,"triangles":len(v)//3,"opening_triangles_added":0,
            "limitations":["synthetic occupancy profiles","facade-aligned box filter reference only, not perspective/GPU anti-aliasing","no measured temporal flight or LOD shader"]}
    (OUT/"night_report.json").write_text(json.dumps(report,indent=2))
    print("NIGHT_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((960,800));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=65,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"NIGHT_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

    # Synthetic spatial metadata stress. Counts correspond to rectangular
    # shell archetypes, not allocated/rendered geometry for the large scenes.
    rows=[]
    for count in (1000,10000,100000):
        i=np.arange(count,dtype=np.int64)
        n=int(np.ceil(np.sqrt(count)))
        xy=np.column_stack((i%n,i//n)).astype(np.float64)*22.
        widths=11+2*(i%3);depths=8+i%2
        upper=xy+np.column_stack((widths,depths))
        centers=(xy+upper)*.5
        tri=np.where(i%4==0,16,12)
        records=np.where(i%4==0,7,6)
        # Match original fixture's wall/roof family indices; base family shared.
        wall=i%4;roof=i%3
        for size in (64,128,256,512,1024,2048):
            start=time.perf_counter()
            cells=np.floor(centers/size).astype(np.int64)
            keys,owner=np.unique(cells,axis=0,return_inverse=True)
            owner=owner.reshape(-1)
            groups=len(keys)
            mins=np.full((groups,2),np.inf);maxs=np.full((groups,2),-np.inf)
            np.minimum.at(mins,owner,xy);np.maximum.at(maxs,owner,upper)
            assert np.all(xy>=mins[owner]) and np.all(upper<=maxs[owner])
            per=np.bincount(owner,minlength=groups)
            counts=np.zeros(groups,dtype=np.int64);np.add.at(counts,owner,tri)
            rec=np.zeros(groups,dtype=np.int64);np.add.at(rec,owner,records)
            assert per.sum()==count and counts.sum()==tri.sum()
            assert rec.sum()==records.sum() and rec.max()<=65536
            mat_groups=int(groups+len(np.unique(owner*4+wall))+len(np.unique(owner*3+roof)))
            assert mat_groups>=3*groups
            # Axis-aligned observation rectangle proxy; not engine frustum/occlusion.
            proxies=[]
            for frac in (.2,.5,.8):
                c=centers.min(0)+(centers.max(0)-centers.min(0))*frac
                lo=c-250;hi=c+250
                direct=np.all(upper>=lo,axis=1)&np.all(xy<=hi,axis=1)
                visible=np.all(maxs>=lo,axis=1)&np.all(mins<=hi,axis=1)
                assert np.all(visible[owner[direct]])
                proxies.append({"direct_triangles":int(tri[direct].sum()),
                                "tile_submitted_triangles":int(counts[visible].sum()),
                                "visible_tiles":int(visible.sum())})
            rows.append({"buildings":count,"tile_m":size,"tile_objects":groups,
                         "per_material_surfaces":mat_groups,"shared_role_surfaces":3*groups,
                         "max_records_per_tile":int(rec.max()),"triangles_model":int(tri.sum()),
                         "parameter_bytes_64B_per_record":int(records.sum()*64),
                         "proxy_views":proxies,"cpu_metadata_ms":(time.perf_counter()-start)*1000})
    spatial={"rows":rows,"scope":"synthetic rectangular AABB metadata model; no geometry allocation at scale, GPU/DIP/SDK not measured",
             "ownership_unique":True,"conservative_proxy_visibility":True}
    (OUT/"spatial_report.json").write_text(json.dumps(spatial,indent=2))
    print("SPATIAL_REPORT "+json.dumps(spatial),flush=True)

if __name__=="__main__":main()
