"""Research 17: shared alpha cutout strip geometry and real EGL discard.
Consumes the exact stage-16 GLB and table; separate cutout layer, not a
production PCGF extension. GLB texture/UV readback is the render input.
"""
from pathlib import Path
from collections import defaultdict
import base64,io,json,math,struct,hashlib
import numpy as np
from PIL import Image,ImageDraw
import trimesh
import moderngl
from render_zero_facade_glsl import raw_mesh
from compile_zero_geometry_tile import y_up,z_up
import zero_facade_architecture as arch
import zero_roof_budgets as roof
from pcg.footprint import Footprint
from pcg.massing import build_massing

OUT=Path("artifacts/zero-alpha-cutout")
KINDS=("roof","balcony","lattice")
PERIOD={"roof":2.4,"balcony":.48,"lattice":.65}
RGB={"roof":(70,73,75),"balcony":(65,49,42),"lattice":(85,89,92)}

def pattern(kind,n):
    # Original analytic repeatable masks. Supersample to retain thin strokes
    # at base resolution; vertical axis is clamped, horizontal axis repeats.
    s=n*4
    x,y=np.meshgrid((np.arange(s)+.5)/s,1-(np.arange(s)+.5)/s)
    def periodic(u):return np.abs((u+.5)%1-.5)
    if kind=="roof":
        m=(periodic(x)<.018/2.4)|(np.abs(y-.53)<.020)|(np.abs(y-.974)<.020)
    elif kind=="balcony":
        m=(periodic(3*x+.12*np.sin(2*np.pi*y))<.055)|(y<.045)|(y>.955)
    else:
        m=(periodic(x)<.035)|(periodic(x+4*y)<.035)|(periodic(x-4*y)<.035)|(periodic(4*y)<.035)
    a=m.reshape(n,4,n,4).mean(axis=(1,3))
    rgba=np.empty((n,n,4),np.uint8);rgba[:,:,:3]=RGB[kind]
    rgba[:,:,3]=np.rint(255*a).astype(np.uint8)
    return rgba

def mip_chain(rgba,correct):
    # Coverage-preservation experiment: global scalar per mip chosen by
    # bisection; algorithm class described by Castano / DirectXTex.
    # Unscaled box-filter pyramid is shared by both lanes. Do not feed
    # rescaled alpha into the next level. Quantization limits tiny levels.
    target=float((rgba[:,:,3]>=128).mean())
    a=rgba[:,:,3].astype(float)/255
    rows=[rgba.copy()];metrics=[dict(size=len(a),coverage=target,scale=1.)]
    while len(a)>1:
        n=len(a)//2;a=a.reshape(n,2,n,2).mean(axis=(1,3))
        def cov(k):return float((np.rint(np.clip(a*k,0,1)*255)>=128).mean())
        lo,hi=0.,64.
        for _ in range(24):
            mid=(lo+hi)/2
            if cov(mid)<target:lo=mid
            else:hi=mid
        k=min((1.,lo,hi),key=lambda q:(abs(cov(q)-target),abs(q-1))) if correct else 1.
        t=np.empty((n,n,4),np.uint8);t[:,:,:3]=rgba[0,0,:3]
        t[:,:,3]=np.rint(np.clip(a*k,0,1)*255).astype(np.uint8)
        rows.append(t);metrics.append(dict(size=n,coverage=cov(k),scale=k))
    return rows,metrics

def strip(points,height,period,closed=False):
    p=np.asarray(points,float);assert p.ndim==2 and p.shape[1]==3
    assert np.isfinite(p).all() and height>0 and period>0
    if closed:p=np.concatenate((p,p[:1]))
    lengths=np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)
    assert np.all(lengths>1e-6)
    d=np.r_[0.,np.cumsum(lengths)]
    # Whole repeat count at closed seam; same phase on adjacent segments.
    actual=d[-1]/max(1,round(d[-1]/period)) if closed else period
    u=d/actual;v=[];uv=[];f=[]
    for i in range(len(lengths)):
        a,b=p[i:i+2];j=len(v)
        v.extend((a,b,b+[0,0,height],a+[0,0,height]))
        uv.extend(((u[i],0),(u[i+1],0),(u[i+1],1),(u[i],1)))
        f.extend(((j,j+1,j+2),(j,j+2,j+3)))
    v=np.asarray(v,np.float32);uv=np.asarray(uv,np.float32);f=np.asarray(f,np.uint32)
    assert np.all(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)>1e-8)
    assert len(f)==2*len(lengths)
    for i in range(len(lengths)-1):
        np.testing.assert_array_equal(v[4*i+1],v[4*(i+1)])
        assert uv[4*i+1,0]==uv[4*(i+1),0]
    if closed:
        np.testing.assert_array_equal(v[0],v[-3])
        assert abs(float(uv[-3,0])-round(float(uv[-3,0])))<1e-5
    return v,uv,f,dict(length_m=float(d[-1]),repeat_m=float(actual),segments=len(lengths),closed=closed)

def join(parts):
    v=[];u=[];f=[];off=0
    for a,b,c,*_ in parts:
        v.append(a);u.append(b);f.append(c+off);off+=len(a)
    return np.concatenate(v),np.concatenate(u),np.concatenate(f)

def chain(edges):
    # Existing planar side skins share exact corner coordinates in GLB.
    def key(p):return tuple(np.round(p.astype(float),5))
    graph=defaultdict(list);coords={}
    for a,b in edges:
        x,y=key(a),key(b);assert x!=y
        coords[x]=a;coords[y]=b;graph[x].append(y);graph[y].append(x)
    assert all(1<=len(x)<=2 for x in graph.values())
    ends=sorted(k for k,v in graph.items() if len(v)==1)
    assert len(ends) in (0,2)
    closed=not ends;start=ends[0] if ends else min(graph)
    path=[start];prev=None
    while True:
        candidates=sorted(q for q in graph[path[-1]] if q!=prev)
        if not candidates:break
        q=candidates[0]
        if q==start:break
        assert q not in path
        prev=path[-1];path.append(q)
    assert len(path)==len(graph) and len(edges)==len(path)-(not closed)
    return np.array([coords[k] for k in path]),closed

def source_geometry():
    root=Path("artifacts/zero-recessed-loggias/depth_1.0")
    table=json.loads((root/"parameters.json").read_text())
    v,n,u,c,f=raw_mesh(root/"tile.glb")
    ids=c[:,0].astype(np.uint32)+256*c[:,1].astype(np.uint32)
    assert np.all(ids[f]==ids[f][:,:1])
    face_ids=ids[f[:,0]]
    placement=json.loads((root/"placements.json").read_text())
    guard_ids={m["id"] for m in placement if m["kind"]=="roof_guard"}
    guard=np.array([table[int(i)].get("placement") in guard_ids for i in face_ids])
    assert guard.sum()==912
    rails=defaultdict(list)
    for r in table:
        if r.get("accessory")=="rail":
            tris=v[f[face_ids==r["id"]]]
            assert len(tris)==2
            verts=np.unique(tris.reshape(-1,3),axis=0)
            lo,hi=verts[:,2].min(),verts[:,2].max()
            bottom=verts[np.abs(verts[:,2]-lo)<1e-6];assert len(bottom)==2
            rails[r["placement"]].append((bottom,hi-lo))
    parts={k:[] for k in KINDS};manifest=[]
    for case in arch.fixtures():
        w=case["outer"][1][0];d=case["outer"][2][1];h=case["eave_m"];off=np.asarray(case["offset"])
        flat=case["roof"]=="flat"
        foot=Footprint(case["building_id"],case["outer"],case["holes"])
        rt=roof.triangles(build_massing(foot,h,case["roof"],case["rise"]).shell)
        paths=[[(.1,.1),(w-.1,.1),(w-.1,d-.1),(.1,d-.1)]] if flat else [
            [(.18,.20),(w-.18,.20)],[(.18,d-.20),(w-.18,d-.20)]]
        for j,path in enumerate(paths):
            p=np.array([(x,y,h+.465 if flat else roof.roof_z(rt,x,y)-.015) for x,y in path])+off
            q=strip(p,.86,PERIOD["roof"],flat)
            parts["roof"].append(q);manifest.append(dict(id=case["building_id"]+":cutout_guard:"+str(j),kind="roof",**q[3]))
        # Every replacement lies inside the measured XY support footprint;
        # roof-plane base is matched to the stage-13/15 recipe.
        old=v[f[guard & np.array([table[int(i)]["building"]==case["building_id"] for i in face_ids])]].reshape(-1,3)
        new=np.concatenate([q[0] for q in parts["roof"][-len(paths):]])
        assert np.all(new[:,:2].min(0)>=old[:,:2].min(0)-1e-5)
        assert np.all(new[:,:2].max(0)<=old[:,:2].max(0)+1e-5)
        assert abs(float(new[:,2].min()-old[:,2].min()))<.03
    assert sum(len(q[2]) for q in parts["roof"])==20
    for mid,entries in sorted(rails.items()):
        h=entries[0][1];assert all(abs(z-h)<1e-6 for _,z in entries)
        path,closed=chain([tuple(e) for e,_ in entries])
        q=strip(path,float(h),PERIOD["balcony"],closed)
        parts["balcony"].append(q);manifest.append(dict(id=mid,kind="balcony",**q[3]))
    railmask=np.array([table[int(i)].get("accessory")=="rail" for i in face_ids])
    assert len(parts["balcony"])>0
    assert railmask.sum()==sum(len(q[2]) for q in parts["balcony"])
    assert not np.any(guard & railmask)
    # Isolated circular lattice fixtures; no antenna physics or arbitrary
    # structural claims. Explicit chord error and continuous UV seam.
    for j,N in enumerate((4,8,12,16)):
        theta=np.arange(N)*2*np.pi/N;r=.5
        p=np.column_stack((j*2+r*np.cos(theta),r*np.sin(theta),np.zeros(N)))
        q=strip(p,3.,PERIOD["lattice"],True);parts["lattice"].append(q)
        manifest.append(dict(id="synthetic_lattice_"+str(N),kind="lattice",radial_error_m=r*(1-math.cos(math.pi/N)),**q[3]))
    return parts,manifest,v,f,face_ids,table,guard,railmask

def export(path,meshes,images):
    scene=trimesh.Scene()
    for kind,(v,u,f) in meshes.items():
        mat=trimesh.visual.material.PBRMaterial(name="cutout_"+kind,
             baseColorTexture=Image.fromarray(images[kind]),baseColorFactor=[1.,1.,1.,1.],
             alphaMode="MASK",alphaCutoff=.5,doubleSided=True,roughnessFactor=.65,metallicFactor=.15)
        m=trimesh.Trimesh(vertices=y_up(v),faces=f,process=False,
             visual=trimesh.visual.texture.TextureVisuals(uv=u,material=mat))
        scene.add_geometry(m,geom_name=kind,node_name=kind)
    def post(t):
        # One spatial batch; shared material/image per pattern, never per
        # individual building. Explicit sampler repeat U and clamp V.
        p=[p for m in t["meshes"] for p in m["primitives"]]
        t["meshes"]=[dict(name="cutout_batch",primitives=p)]
        t["nodes"]=[dict(name="cutout_batch",mesh=0)];t["scenes"]=[dict(nodes=[0])];t["scene"]=0
        t["samplers"]=[dict(wrapS=10497,wrapT=33071,minFilter=9987,magFilter=9729)]
        for x in t["textures"]:x["sampler"]=0
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene,include_normals=True,tree_postprocessor=post))

def read(path,expected,images):
    b=path.read_bytes();magic,ver,size=struct.unpack_from("<4sII",b)
    assert (magic,ver,size)==(b"glTF",2,len(b))
    length,tag=struct.unpack_from("<I4s",b,12);assert tag==b"JSON"
    t=json.loads(b[20:20+length]);bn,tag=struct.unpack_from("<I4s",b,20+length);assert tag==b"BIN\0"
    blob=b[28+length:28+length+bn];assert len(blob)==bn
    assert len(t["nodes"])==len(t["meshes"])==1
    assert len(t["images"])==len(t["materials"])==len(expected)
    def view(i):
        d=t["bufferViews"][i];a=d.get("byteOffset",0);return blob[a:a+d["byteLength"]]
    def get(i):
        a=t["accessors"][i];width={"SCALAR":1,"VEC2":2,"VEC3":3}[a["type"]]
        dt=np.dtype({5123:"<u2",5125:"<u4",5126:"<f4"}[a["componentType"]])
        return np.frombuffer(view(a["bufferView"]),dtype=dt,count=a["count"]*width,
                              offset=a.get("byteOffset",0)).reshape(-1,width)
    out={}
    for p in t["meshes"][0]["primitives"]:
        mat=t["materials"][p["material"]];kind=mat["name"].removeprefix("cutout_")
        assert kind not in out
        assert mat["alphaMode"]=="MASK" and mat["alphaCutoff"]==.5 and mat["doubleSided"]
        tex=t["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        sampler=t["samplers"][tex["sampler"]];assert sampler["wrapS"]==10497 and sampler["wrapT"]==33071
        im=t["images"][tex["source"]];a=np.asarray(Image.open(io.BytesIO(view(im["bufferView"]))).convert("RGBA"))
        np.testing.assert_array_equal(a,images[kind])
        v=z_up(get(p["attributes"]["POSITION"]));u=get(p["attributes"]["TEXCOORD_0"])
        f=get(p["indices"]).reshape(-1,3).astype(np.uint32)
        ev,eu,ef=expected[kind]
        np.testing.assert_array_equal(v,ev);np.testing.assert_array_equal(f,ef)
        np.testing.assert_allclose(u,np.column_stack((eu[:,0],1-eu[:,1])),atol=1e-7,rtol=0)
        out[kind]=(v,u,f,a)
    assert out.keys()==expected.keys()
    return out

VERT="""#version 330
in vec3 p; in vec2 uv; out vec2 tc;
uniform vec3 center,right,up,view;
uniform float scale;
void main(){vec3 q=p-center;tc=uv;gl_Position=vec4(dot(q,right)*scale,dot(q,up)*scale,-dot(q,view)/200.,1.);}
"""
FRAG="""#version 330
in vec2 tc;out vec4 color;
uniform sampler2D mask_tex;uniform bool masked;uniform vec3 tint;
void main(){if(masked){vec4 c=texture(mask_tex,tc);if(c.a<0.5)discard;color=vec4(c.rgb,1.);}
else color=vec4(tint,1.);}
"""

class Lab:
    def __init__(self):
        self.ctx=moderngl.create_context(standalone=True,backend="egl",require=330)
        assert any(x in self.ctx.info["GL_RENDERER"].lower() for x in ("llvmpipe","softpipe","swrast","software"))
        self.p=self.ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
        self.p["mask_tex"].value=0
        self.color=self.ctx.texture((640,640),4)
        self.depth=self.ctx.depth_texture((640,640))
        self.fbo=self.ctx.framebuffer([self.color],self.depth)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND|moderngl.CULL_FACE)
    def mesh(self,v,u,f):
        a=self.ctx.buffer(np.column_stack((v,u)).astype("f4").tobytes())
        b=self.ctx.buffer(f.astype("u4").tobytes())
        q=self.ctx.vertex_array(self.p,[(a,"3f 2f","p","uv")],index_buffer=b)
        return a,b,q
    def texture(self,levels,anisotropy=1.):
        n=len(levels[0]);t=self.ctx.texture((n,n),4,levels[0].tobytes())
        t.build_mipmaps(0,len(levels)-1)
        for i,a in enumerate(levels):t.write(a.tobytes(),level=i)
        t.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR)
        t.repeat_x=True;t.repeat_y=False;t.anisotropy=anisotropy
        return t
    def render(self,objects,center,ppm,view):
        self.fbo.use();self.fbo.clear(.86,.89,.91,1.,depth=1.)
        v=np.asarray(view,float);v/=np.linalg.norm(v)
        r=np.cross([0,0,1],v);r/=np.linalg.norm(r);u=np.cross(v,r)
        for k,x in (("center",center),("right",r),("up",u),("view",v)):self.p[k].value=tuple(x)
        self.p["scale"].value=2*ppm/640
        for mesh,tex,tint in objects:
            self.p["masked"].value=tex is not None;self.p["tint"].value=tint
            if tex is not None:tex.use(0)
            mesh[2].render(moderngl.TRIANGLES)
        im=np.frombuffer(self.fbo.read(components=3,alignment=1),np.uint8).reshape(640,640,3)[::-1].copy()
        dep=np.frombuffer(self.depth.read(),np.float32).reshape(640,640)[::-1].copy()
        return im,dep

def depth_test(lab,tex):
    v,u,f,_=strip([(0,0,0),(2.4,0,0)],.86,2.4)
    u[:,1]=1-u[:,1];front=lab.mesh(v,u,f);back=lab.mesh(v+[0,.3,0],u,f)
    cam=((1.2,0,.43),200.,(0,-1,0))
    _,d0=lab.render([(back,None,(.8,.2,.1))],*cam)
    _,df=lab.render([(front,tex,(0,0,0))],*cam)
    im,d=lab.render([(front,tex,(0,0,0)),(back,None,(.8,.2,.1))],*cam)
    mask=d0<1;bars=df<1;holes=mask&~bars
    assert bars.sum()>100 and holes.sum()>100 and not np.any(bars&~mask)
    np.testing.assert_array_equal(d[bars],df[bars])
    np.testing.assert_array_equal(d[holes],d0[holes])
    assert np.all(d[bars]<d0[bars])
    _,rear=lab.render([(front,tex,(0,0,0))],(1.2,0,.43),200.,(0,1,0))
    assert abs(int((rear<1).sum())-int(bars.sum()))<=2
    Image.fromarray(im).save(OUT/"discard_depth.png")
    return dict(front_pixels=int(bars.sum()),through_hole_pixels=int(holes.sum()),backface_pixels=int((rear<1).sum()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parts,manifest,v,f,ids,table,guard,rail=source_geometry()
    images={k:pattern(k,256) for k in KINDS}
    meshes={k:join(parts[k]) for k in KINDS}
    path=OUT/"cutout_batch.glb";export(path,meshes,images);actual=read(path,meshes,images)
    (OUT/"placements.json").write_text(json.dumps(manifest,indent=2))
    lab=Lab();objects={};mip_metrics={}
    for k,(a,u,b,img) in actual.items():
        m=lab.mesh(a,u,b);objects[k]=m
        for n in (128,256,512):
            rgba=img if n==256 else pattern(k,n)
            Image.fromarray(rgba).save(OUT/f"{k}_{n}.png")
            for correct in (False,True):
                levels,metrics=mip_chain(rgba,correct)
                mip_metrics[f"{k}_{n}_{correct}"]=metrics
                if correct:
                    if n==256:
                        directory=OUT/"mips"/k;directory.mkdir(parents=True,exist_ok=True)
                        for level,image in enumerate(levels):Image.fromarray(image).save(directory/f"{level}.png")
                    std=mip_metrics[f"{k}_{n}_False"];target=metrics[0]["coverage"]
                    assert all(abs(q["coverage"]-target)<=abs(s["coverage"]-target)+1e-12 for q,s in zip(metrics,std))
    textures={k:lab.texture(mip_chain(actual[k][3],True)[0]) for k in KINDS}
    depth=depth_test(lab,textures["roof"])
    # Actual stage-16 scene geometry, neutral color for context; the
    # existing facade shader is NOT replaced by this diagnostic.
    opaque=lab.mesh(v,np.zeros((len(v),2)),f[~(guard|rail)])
    oldguard=lab.mesh(v,np.zeros((len(v),2)),f[guard])
    oldrail=lab.mesh(v,np.zeros((len(v),2)),f[rail])
    base=[(opaque,None,(.65,.64,.59))]
    cams=[((61,6,10),4.3,(.35,-1,.45)),((58,5,21),28.,(.25,-1,.42)),
          ((107,0,9),32.,(.25,-1,.16)),((107,6,16),24.,(.65,-1,1.8))]
    sheet=Image.new("RGB",(1280,640*len(cams)),(220,225,230))
    for j,cam in enumerate(cams):
        for i,lane in enumerate(("geometry","cutout")):
            extra=[(oldguard,None,(.27,.28,.29)),(oldrail,None,(.25,.19,.16))] if not i else [
                (objects[k],textures[k],(0,0,0)) for k in ("roof","balcony")]
            im,_=lab.render(base+extra,*cam);im=Image.fromarray(im)
            im.save(OUT/f"{lane}_{j}.png");ImageDraw.Draw(im).text((8,8),lane,fill=(0,0,0));sheet.paste(im,(i*640,j*640))
    sheet.save(OUT/"comparison.png")
    im,_=lab.render([(objects["lattice"],textures["lattice"],(0,0,0))],(3,0,1.5),80.,(.4,-1,.18))
    Image.fromarray(im).save(OUT/"lattice.png")
    # Fixed projected pixel scale and oblique/moving-camera coverage probe.
    # Count foreground pixels normalized by opaque support pixel count.
    temporal=[]
    for n in (128,256,512):
        rgba=pattern("roof",n)
        for method in ("standard","coverage","coverage_aniso"):
            tex=lab.texture(mip_chain(rgba,method!="standard")[0],8. if method.endswith("aniso") else 1.)
            a,u,b,_=strip([(0,0,0),(9.6,0,0)],.86,PERIOD["roof"]);u[:,1]=1-u[:,1]
            m=lab.mesh(a,u,b)
            for angle in (0,30,60,80):
                view=(math.sin(math.radians(angle)),-math.cos(math.radians(angle)),0.)
                for ppm in (7.,25.,60.):
                    values=[];previous=None;xor=[]
                    for j in range(5):
                        center=(4.8+j*.2/ppm,0,.43)
                        _,full=lab.render([(m,None,(1,1,1))],center,ppm,view)
                        _,cut=lab.render([(m,tex,(0,0,0))],center,ppm,view)
                        visible=cut<1;total=int((full<1).sum());assert total>0
                        assert not np.any(visible & ~(full<1))
                        values.append(float(visible.sum()/total))
                        if previous is not None:xor.append(int(np.count_nonzero(visible!=previous)))
                        previous=visible
                    temporal.append(dict(resolution=n,method=method,angle_deg=angle,ppm=ppm,
                         post_width_px=.036*ppm*math.cos(math.radians(angle)),
                         coverage_mean=float(np.mean(values)),coverage_std=float(np.std(values)),
                         adjacent_mask_xor_mean=float(np.mean(xor))))
            for resource in reversed(m):resource.release()
            tex.release()
    result=dict(schema="pcg.research.cutout.v1",renderer=lab.ctx.info["GL_RENDERER"],
        source_triangles=len(f),roof_before=int(guard.sum()),roof_after=len(meshes["roof"][2]),
        balcony_before=int(rail.sum()),balcony_after=len(meshes["balcony"][2]),
        candidate_scene_triangles=int(len(f)-guard.sum()-rail.sum()+len(meshes["roof"][2])+len(meshes["balcony"][2])),
        cutout_glb_bytes=path.stat().st_size,materials=len(actual),depth_test=depth,
        source_cells_sha256=hashlib.sha256(Path("artifacts/zero-recessed-loggias/depth_1.0/facade_cells.bin").read_bytes()).hexdigest(),
        mip_metrics=mip_metrics,temporal=temporal,limitations=[
        "Separate shared cutout layer; opaque context rendered from stage-16 GLB with neutral tint, not full facade/PBR shader.",
        "GLB embeds base masks only; explicit experimental mip chains are supplied to EGL separately, not glTF runtime assets.",
        "Open rail sides only; opaque balcony cladding and partial lower panels retained.",
        "Lattice cylindrical fixtures are synthetic, no antenna functional simulation.",
        "Software EGL fragment/depth evidence, not hardware timing or UNIGINE shadow/TAA/MSAA acceptance.",
        "Temporal metric is unregistered adjacent mask XOR, not motion-compensated shimmer or measured GPU overdraw.",
        "No collision geometry or structural equivalence claimed."])
    (OUT/"report.json").write_text(json.dumps(result,indent=2))
    print("CUTOUT_REPORT "+json.dumps({k:v for k,v in result.items() if k not in ("mip_metrics","temporal")}),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=76,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"CUTOUT_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main()
