"""Fixed-camera CPU depth-buffer views of actual exported L2 GLBs."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pcg.footprint import Footprint
from pcg.massing import build_massing

VIEWS=[("front",[0,-1,0],[0,0,1]),("side",[1,0,0],[0,0,1]),
       ("top",[0,0,1],[0,1,0]),("iso",[1,-1,0.8],[0,0,1])]
SIZE=384
BG=(245,243,238)


def raster(mesh,view,up,*,textured=False,night=False):
    view=np.array(view,dtype=float);view/=np.linalg.norm(view)
    right=np.cross(up,view);right/=np.linalg.norm(right)
    up=np.cross(view,right)
    v=mesh.vertices-np.array([6,5,4])
    xy=np.column_stack((v@right,-v@up))*20+SIZE/2
    z=v@view
    img=np.empty((SIZE,SIZE,3),dtype=np.uint8);img[:]=BG
    depth=np.full((SIZE,SIZE),-np.inf)
    light=np.array([0.4,-0.6,0.7]);light/=np.linalg.norm(light)
    for k,face in enumerate(mesh.faces):
        if mesh.face_normals[k]@view<=1e-8:continue
        a,b,c=xy[face]
        lo=np.maximum(np.floor(np.min(xy[face],axis=0)).astype(int),0)
        hi=np.minimum(np.ceil(np.max(xy[face],axis=0)).astype(int)+1,SIZE)
        x0,y0=lo;x1,y1=hi
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];xx=xx+0.5;yy=yy+0.5
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        wc=1-wa-wb
        dz=wa*z[face[0]]+wb*z[face[1]]+wc*z[face[2]]
        dst=depth[y0:y1,x0:x1]
        mask=(wa>=-1e-8)&(wb>=-1e-8)&(wc>=-1e-8)&(dz>dst)
        dst[mask]=dz[mask]
        shade=0.68+0.32*max(0.,mesh.face_normals[k]@light)
        if textured:
            tex=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'))
            uv=mesh.visual.uv[face]
            sample=wa[mask,None]*uv[0]+wb[mask,None]*uv[1]+wc[mask,None]*uv[2]
            tx=np.floor((sample[:,0]%1)*tex.shape[1]).astype(int)%tex.shape[1]
            ty=np.floor((-sample[:,1]%1)*tex.shape[0]).astype(int)%tex.shape[0]
            rgb=tex[ty,tx]
        else:
            rgb=mesh.visual.vertex_colors[face,:3].mean(axis=0)
        if night:
            if not textured:raise ValueError("night diagnostic requires PBR materials")
            srgb=rgb.astype(np.float64)/255
            linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
            factor=mesh.visual.material.emissiveFactor
            emission=np.zeros(3) if factor is None else np.asarray(factor)
            linear=np.clip(linear*shade*.055+emission,0,1)
            rgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)*255
            img[y0:y1,x0:x1][mask]=np.uint8(np.round(np.clip(rgb,0,255)))
        else:
            img[y0:y1,x0:x1][mask]=np.uint8(rgb*shade)
    if not textured:
        assert np.isfinite(depth).sum()>100
    return Image.fromarray(img),depth


def main():
    out=Path("artifacts/l2-massing");out.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path("tests/fixtures/l2_massing.json").read_text())
    report=[]
    grid=Image.new("RGB",(4*SIZE,len(cases)*SIZE),"white")
    for row,c in enumerate(cases):
        name=c["id"]
        f=Footprint(name,c["outer"],c.get("holes",[]))
        result=build_massing(f,6,c["roof"],c.get("rise",0))
        s=result.shell
        triangles=s.triangles()
        mesh=trimesh.Trimesh(vertices=s.vertices,faces=[t for _,t in triangles],process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent
        volume=float(mesh.volume)
        roles={x.id:x.role for x in s.surfaces}
        palette={"base":[110,110,110,255],"wall":[218,205,180,255],"roof":[150,75,55,255]}
        colors=np.array([palette[roles[k]] for k,_ in triangles],dtype=np.uint8)
        bounds=mesh.bounds.copy()
        mesh.unmerge_vertices()
        mesh.visual.vertex_colors=np.repeat(colors,3,axis=0)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
        path=out/(name+".glb");path.write_bytes(mesh.export(file_type="glb"))
        loaded=trimesh.load(path,force="mesh",process=False)
        loaded.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        np.testing.assert_allclose(loaded.bounds,bounds,atol=2e-6)
        assert len(loaded.faces)==len(triangles)
        np.testing.assert_allclose(loaded.volume,volume,rtol=1e-6)
        sheet=Image.new("RGB",(2*SIZE,2*SIZE),"white")
        for i,(label,view,up) in enumerate(VIEWS):
            canvas,depth=raster(loaded,view,up)
            if label=="top":
                # Orthographic roof coverage: pixel-center samples well away
                # from boundaries must agree with the canonical footprint.
                from shapely import contains_xy, distance, points
                yy,xx=np.mgrid[0:SIZE,0:SIZE]
                x=(xx+0.5-SIZE/2)/20+6
                y=-(yy+0.5-SIZE/2)/20+5
                poly=f.polygon()
                expected=contains_xy(poly,x,y)
                safe=distance(poly.boundary,points(x,y))>0.06
                assert not np.any(safe & (expected!=np.isfinite(depth))), name
            ImageDraw.Draw(canvas).text((10,10),name+" / "+label,fill=(0,0,0))
            canvas.save(out/(name+"_"+label+".png"))
            sheet.paste(canvas,((i%2)*SIZE,(i//2)*SIZE))
            grid.paste(canvas,(i*SIZE,row*SIZE))
        sheet.save(out/(name+"_contact.png"))
        (out/(name+".json")).write_text(json.dumps({"schema_version":1,"compiler_version":"0.1.0","input_digest":f.digest(),"massing":asdict(result)},sort_keys=True,indent=2))
        report.append({"id":name,"requested":result.requested_roof,"actual":result.actual_roof,
                       "fallback":result.fallback_reason,"area_m2":f.polygon().area,
                       "volume_m3":volume,"triangles":len(triangles),"holes":len(f.holes)})
    grid.save(out/"all_contact.png")
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=="__main__":main()
