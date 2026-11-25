"""Condor shared-atlas UV experiment; original PCG procedural pixels, not Condor DDS."""
from pathlib import Path
import base64
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from pcg.materials import MATERIALS, texture_maps
from render_l2_massing import raster
from benchmark_condor import CASES, run, REVISION, roof_coverage
from condor_buildings.processing.mesh_grouper import MeshGrouper

OUT=Path("artifacts/condor-textured")
OUT.mkdir(parents=True,exist_ok=True)

def pattern(name,size,seed):
    a=texture_maps(MATERIALS[name],256,seed)["basecolor"]
    return Image.fromarray(np.uint8(np.clip(a*255,0,255))).resize(size,Image.Resampling.LANCZOS)

def window(draw,x,y,w,h,door=False):
    draw.rectangle((x-3,y-3,x+w+3,y+h+3),fill=(119,112,100))
    draw.rectangle((x-1,y-1,x+w+1,y+h+1),fill=(221,215,196))
    draw.rectangle((x,y,x+w,y+h),fill=(55,65,69) if not door else (92,70,52))
    if not door:
        for j in range(h):
            a=j/max(h,1)
            col=tuple(int(c*(1-a)+b*a) for c,b in zip((119,142,154),(38,48,52)))
            draw.line((x+1,y+j,x+w-1,y+j),fill=col)
        draw.line((x+w//2,y,x+w//2,y+h),fill=(209,208,192),width=2)
        draw.line((x,y+h//3,x+w,y+h//3),fill=(192,193,181),width=2)
    else:
        draw.rectangle((x+4,y+5,x+w-4,y+h//2),fill=(61,78,84))
        draw.ellipse((x+w-7,y+h//2,x+w-4,y+h//2+3),fill=(205,180,113))
    draw.rectangle((x-4,y+h+2,x+w+4,y+h+5),fill=(166,160,146))

def row(w,h,index,section,high=False,industrial=False):
    names=("plaster_cream","plaster_white","red_brick","concrete_panel")
    name=names[(index+ (2 if industrial else 0))%len(names)]
    tile=pattern(name,(128,128),index+17)
    im=Image.new("RGB",(w,h))
    for y in range(0,h,128):
        for x in range(0,w,128): im.paste(tile,(x,y))
    d=ImageDraw.Draw(im)
    if section=="gable": return im
    if section=="ground":
        d.rectangle((0,h-11,w,h),fill=(104,98,88))
    d.line((0,0,w,0),fill=(166,160,148),width=3)
    bays=8 if high else 3
    for i in range(bays):
        cx=int((i+.5)*w/bays)
        door=section=="ground" and i==0
        ww=int(w/bays*(.48 if not industrial else .68))
        hh=int(h*(.69 if door else .52))
        y=h-hh-9 if door else int(h*.20)
        window(d,cx-ww//2,y,ww,hh,door)
    return im

def atlases():
    maps={}
    houses=Image.new("RGB",(256,6144))
    roofs=("ceramic_roof","metal_roof_red","bitumen_roof","ceramic_roof","metal_roof_red","bitumen_roof")
    for i,name in enumerate(roofs):
        houses.paste(pattern(name,(256,256),30+i),(0,i*256))
    for i in range(12):
        for j,sec in enumerate(("gable","upper","ground")):
            houses.paste(row(256,128,i,sec),(0,1536+i*384+j*128))
    maps["houses"]=houses
    industrial=houses.copy()
    for i in range(12):
        for j,sec in enumerate(("gable","upper","ground")):
            industrial.paste(row(256,128,i,sec,industrial=True),(0,1536+i*384+j*128))
    maps["industrial_walls"]=industrial
    high=Image.new("RGB",(1024,6144))
    for i in range(12):
        for j in range(4):
            high.paste(row(1024,128,i,"ground" if j==3 else "upper",high=True),(0,i*512+j*128))
    maps["Highrise_walls"]=high
    for i in range(6):
        maps[f"flat_roof_{i+1}"]=pattern("bitumen_roof",(256,256),90+i)
    for key,im in maps.items():im.save(OUT/(key+".png"))
    return maps

def to_mesh(part,key,maps):
    verts=[];uvs=[]
    assert len(part.faces)==len(part.face_uvs)
    for f,u in zip(part.faces,part.face_uvs):
        for j in range(1,len(f)-1):
            for k in (0,j,j+1):
                verts.append(part.vertices[f[k]-1]);uvs.append(part.uvs[u[k]-1])
    mat=trimesh.visual.material.PBRMaterial(name=key,baseColorTexture=maps[key],metallicFactor=0.,roughnessFactor=.85)
    return trimesh.Trimesh(vertices=np.array(verts),faces=np.arange(len(verts)).reshape(-1,3),process=False,visual=trimesh.visual.texture.TextureVisuals(uv=np.array(uvs),material=mat))

def export(grouper,maps,path):
    scene=trimesh.Scene();expected={}
    for key,p in grouper.get_non_empty_groups().items():
        m=to_mesh(p,key,maps);expected[key]=(m.vertices.copy(),m.visual.uv.copy())
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
        scene.add_geometry(m,geom_name=key,node_name=key)
    path.write_bytes(scene.export(file_type="glb"))
    loaded=trimesh.load(path,force="scene",process=False)
    result=[]
    for m in loaded.geometry.values():
        k=m.visual.material.name
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        np.testing.assert_allclose(m.vertices,expected[k][0],atol=3e-5)
        np.testing.assert_allclose(m.visual.uv,expected[k][1],atol=3e-5)
        np.testing.assert_array_equal(np.asarray(m.visual.material.baseColorTexture.convert("RGB")),np.asarray(maps[k]))
        result.append(m)
    assert len(result)==len(expected)
    return result

def render(meshes,view,center,ppm,size):
    depth=np.full((size,size),-np.inf); canvas=np.full((size,size,3),(245,243,238),dtype=np.uint8)
    for m in meshes:
        im,z=raster(m,view,[0,1,0] if view==[0,0,1] else [0,0,1],textured=True,center=center,pixels_per_m=ppm,size=size)
        mask=z>depth;canvas[mask]=np.asarray(im)[mask];depth[mask]=z[mask]
    assert np.isfinite(depth).sum()>100
    return Image.fromarray(canvas)

def preview(name,im):
    thumb=im.copy();thumb.thumbnail((800,900))
    buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=70,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"CONDOR_{name}_{i//4000:03d} "+data[i:i+4000],flush=True)

def main():
    maps=atlases()
    size=360;sheet=Image.new("RGB",(size*3,size*len(CASES)+28),(245,243,238))
    ImageDraw.Draw(sheet).text((8,8),"CONDOR / original donor UV / shared synthetic test atlases / diffuse only",fill=(0,0,0))
    report={"revision":REVISION,"native_textures":False,"shared_atlases":{k:list(v.size) for k,v in maps.items()},"cases":[]}
    for i,c in enumerate(CASES):
        b,r,_=run(c,0);g=MeshGrouper();g.add_building(b,r)
        meshes=export(g,maps,OUT/(c[0]+".glb"))
        outer=np.asarray(c[5]);lo=outer.min(0);hi=outer.max(0)
        center=(*((lo+hi)/2),c[4]/2+1.5);ppm=(size-40)/(max(hi-lo)+c[4]*.5+4)
        count=sum(len(m.faces) for m in meshes)
        for j,view in enumerate(([1,-1,.8],[0,-1,0],[0,0,1])):
            im=render(meshes,view,center,ppm,size)
            ImageDraw.Draw(im).text((8,8),f"{c[0]} / {count} tris",fill=(0,0,0))
            sheet.paste(im,(j*size,i*size+28))
        report["cases"].append({"name":c[0],"triangles":count,"roof":r.actual_roof_type.value,"coverage":roof_coverage(c,r)})
    sheet.save(OUT/"textured_contact.png");preview("TEXTURED",sheet)
    g=MeshGrouper()
    for i in range(64):
        b,r,_=run(CASES[i%len(CASES)],0,((i%8)*48,(i//8)*48),42+i);g.add_building(b,r)
    meshes=export(g,maps,OUT/"block.glb")
    im=render(meshes,[1,-1,1.4],(185,185,0),1.8,1000)
    ImageDraw.Draw(im).text((8,8),"64 buildings / shared test textures / diffuse only",fill=(0,0,0))
    im.save(OUT/"textured_block.png");preview("BLOCK",im)
    report["block"]={"triangles":sum(len(m.faces) for m in meshes),"groups":len(meshes)}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("CONDOR_TEXTURE_REPORT "+json.dumps(report),flush=True)

if __name__=="__main__":main()
