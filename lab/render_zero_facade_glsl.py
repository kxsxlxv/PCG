"""Actual GLSL 330 execution on Mesa software OpenGL/EGL, not UNIGINE/UUSL.
Read generated GLB and little-endian ABI; shared array textures only.
"""
from pathlib import Path
import base64,io,json,struct,os
import numpy as np
from PIL import Image,ImageDraw
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE","1")
os.environ.setdefault("EGL_PLATFORM","surfaceless")
import moderngl
from pcg.far_emission import srgb_decode,srgb_encode
from pcg._material_patterns import noise
from compile_zero_geometry_tile import lookup,z_up
from compile_zero_tiles import HEADER,RECORD,LAYERS

OUT=Path("artifacts/zero-glsl")
W=1024

def raw_mesh(path):
    raw=path.read_bytes()
    magic,ver,total=struct.unpack_from("<4sII",raw);assert magic==b"glTF" and ver==2 and total==len(raw)
    n=struct.unpack_from("<I",raw,12)[0];tree=json.loads(raw[20:20+n])
    blob=raw[28+n:]
    def get(i):
        a=tree["accessors"][i];v=tree["bufferViews"][a["bufferView"]]
        width={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[a["type"]]
        dt=np.dtype({5121:"u1",5123:"<u2",5125:"<u4",5126:"<f4"}[a["componentType"]])
        assert v.get("byteStride",width*dt.itemsize)==width*dt.itemsize
        return np.frombuffer(blob,dtype=dt,count=a["count"]*width,offset=v.get("byteOffset",0)+a.get("byteOffset",0)).reshape(-1,width)
    vs=[];ns=[];us=[];cs=[];fs=[];off=0
    for p in tree["meshes"][0]["primitives"]:
        a=p["attributes"];v=z_up(get(a["POSITION"]));n=z_up(get(a["NORMAL"]))
        u=get(a["TEXCOORD_1"]);c=get(a["COLOR_0"])
        assert tree["accessors"][a["COLOR_0"]]["normalized"]
        f=get(p["indices"]).reshape(-1,3).astype(np.uint32)
        vs.append(v);ns.append(n);us.append(u);cs.append(c);fs.append(f+off);off+=len(v)
    return tuple(np.ascontiguousarray(np.concatenate(x)) for x in (vs,ns,us,cs,fs))

def module_library(n=256):
    x,y=np.meshgrid((np.arange(n)+.5)/n,1-(np.arange(n)+.5)/n)
    colors=[];nhs=[];masks=[]
    for style in range(6):
        border=np.minimum.reduce([x,1-x,y,1-y])
        frame=border<(.085 if style!=3 else .065)
        if style==0:frame|=np.abs(x-.5)<.035
        if style==1:frame|=(np.abs(x-.5)<.03)|(np.abs(y-.72)<.028)
        if style==2:frame|=(np.abs(x-1/3)<.025)|(np.abs(x-2/3)<.025)
        if style==3:frame|=np.abs(y-.52)<.03
        if style==4:frame|=(y<.16)|(np.abs(y-.52)<.028)
        if style==5:frame[:]=True
        glass=(~frame).astype(float)
        rng=np.random.default_rng(408+style)
        grains=noise(n,24,rng)-.5
        curtain=np.clip(.65+.25*np.sin(x*26+style)+.12*grains,0.15,1)
        glass_color=np.stack([.16+.13*y+.07*curtain,.23+.15*y+.05*curtain,.28+.18*y+.04*curtain],-1)
        frame_color=np.array([.72,.70,.63] if style in (1,4) else [.85,.84,.80])
        if style==5:frame_color=np.array([.35,.24,.16])
        rgb=glass_color*glass[...,None]+frame_color*(1-glass[...,None])
        rgb=np.clip(rgb*(1+.04*grains[...,None]),0,1)
        height=.15+.65*frame.astype(float)+.025*grains
        if style==5:
            panel=((x>.14)&(x<.86)&(((y>.14)&(y<.44))|((y>.55)&(y<.86))))
            height-=.23*panel
            handle=((x-.77)**2+(y-.49)**2)<.002
            rgb[handle]=[.56,.49,.31];height[handle]=.95
        # Gaussian-like small separable blur (edge padded), then metric relief normal.
        for _ in range(2):
            h=np.pad(height,1,mode="edge")
            height=(4*h[1:-1,1:-1]+h[:-2,1:-1]+h[2:,1:-1]+h[1:-1,:-2]+h[1:-1,2:])/8
        gy,gx=np.gradient(height,.055/n,.045/n)
        normal=np.stack([-gx*.0018,gy*.0018,np.ones_like(height)],-1)
        normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
        nh=np.concatenate((normal*.5+.5,height[...,None]),-1)
        mask=np.stack((glass,curtain,.52-.39*glass,.83+.17*(1-glass)),axis=-1)
        d=OUT/"modules"/str(style);d.mkdir(parents=True,exist_ok=True)
        for name,a in (("basecolor",rgb),("normal_height",nh),("mask",mask)):
            Image.fromarray(np.uint8(np.round(np.clip(a,0,1)*255))).save(d/(name+".png"))
        colors.append(srgb_decode(rgb));nhs.append(nh);masks.append(mask)
    return np.array(colors),np.array(nhs),np.array(masks)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    root=Path("artifacts/zero-multitile")
    idx=json.loads((root/"tiles128/index.json").read_text())
    chosen=max(idx["tiles"],key=lambda t:t["buildings"])
    directory=(root/"tiles128"/chosen["manifest"]).parent
    table=json.loads((directory/"parameters.json").read_text())
    cells=np.frombuffer((directory/"facade_cells.bin").read_bytes(),dtype=np.uint8)
    raw=(directory/"facade_params.bin").read_bytes()
    magic,ver,count,stride=HEADER.unpack_from(raw)
    assert (magic,ver,count,stride)==(b"PCGF",1,len(table),RECORD.size)
    records=np.frombuffer(raw,dtype="<u4",offset=16).reshape(count,28)
    u=np.ascontiguousarray(records[:,:12]).reshape(-1,4)
    f=np.ascontiguousarray(records[:,12:]).view("<f4").reshape(-1,4)
    v,n,uv,color,faces=raw_mesh(directory/"tile.glb")
    ids=color[:,0].astype(np.uint32)+256*color[:,1].astype(np.uint32)
    assert np.all(ids[faces]==ids[faces][:,:1])
    ctx=moderngl.create_context(standalone=True,backend="egl",require=330)
    renderer=ctx.info["GL_RENDERER"]
    assert any(x in renderer.lower() for x in ("llvmpipe","softpipe","swrast","software")),renderer
    prog=ctx.program(vertex_shader=Path("lab/shaders/zero_facade.vert").read_text(),
                     fragment_shader=Path("lab/shaders/zero_facade.frag").read_text())
    keep=[]
    def buffer(a):
        b=ctx.buffer(np.ascontiguousarray(a).tobytes());keep.append(b);return b
    vao=ctx.vertex_array(prog,[(buffer(v.astype("f4")),"3f","in_position"),
        (buffer(n.astype("f4")),"3f","in_normal"),(buffer(uv.astype("f4")),"2f","in_layout"),
        (buffer(color.astype("u1")),"4f1","in_color")],index_buffer=buffer(faces.astype("u4")),index_element_size=4)
    def data_tex(name,a,dtype,components,unit):
        a=np.asarray(a).reshape(-1,components);height=max(1,(len(a)+W-1)//W)
        pad=np.zeros((height*W,components),dtype=a.dtype);pad[:len(a)]=a
        t=ctx.texture((W,height),components,pad.tobytes(),dtype=dtype,alignment=1)
        t.filter=(moderngl.NEAREST,moderngl.NEAREST);t.repeat_x=t.repeat_y=False
        t.use(unit);prog[name].value=unit;keep.append(t)
    data_tex("param_u",u,"u4",4,0);data_tex("param_f",f,"f4",4,1);data_tex("cells",cells,"u1",1,2)
    lib=root/"library";manifest=json.loads((lib/"manifest.json").read_text())
    assert [r["material"] for r in manifest["layers"]]==list(LAYERS)
    def array_tex(name,arr,dtype,unit):
        arr=np.asarray(arr)
        if dtype=="f1":arr=np.uint8(np.round(np.clip(arr,0,1)*255))
        else:arr=arr.astype(np.float16)
        layers,h,w,comp=arr.shape
        t=ctx.texture_array((w,h,layers),comp,np.ascontiguousarray(arr).tobytes(),alignment=1,dtype=dtype)
        t.build_mipmaps();t.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR)
        t.repeat_x=t.repeat_y=True
        t.use(unit);prog[name].value=unit;keep.append(t)
    for k,name,dtype,unit in (("basecolor","base_color","f2",3),("normal","base_normal","f1",4),("orm","base_orm","f1",5)):
        arr=np.array([np.asarray(Image.open(lib/r["maps"][k]["path"]),dtype=np.float32)/255 for r in manifest["layers"]])
        if k=="basecolor":arr=srgb_decode(arr)
        array_tex(name,arr,dtype,unit)
    mc,mn,mm=module_library()
    array_tex("module_color",mc,"f2",6);array_tex("module_nh",mn,"f1",7);array_tex("module_mask",mm,"f1",8)
    mask=np.uint8(np.round(noise(256,12,np.random.default_rng(447))*255))
    macro=ctx.texture((256,256),1,mask.tobytes(),dtype="f1");macro.build_mipmaps()
    macro.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR);macro.use(9);prog["macro_mask"].value=9;keep.append(macro)
    size=640
    target=ctx.texture((size,size),4,dtype="f4");depth=ctx.depth_texture((size,size))
    fb=ctx.framebuffer([target],depth);fb.use()
    ctx.enable(moderngl.DEPTH_TEST|moderngl.CULL_FACE)
    view=np.array([1.,-1.5,.85]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
    prog["view_dir"].value=tuple(view)
    lo=v.min(0);hi=v.max(0);dscale=max(100.,2*np.linalg.norm(hi-lo))
    boxes=[]
    for building in sorted({r["building"] for r in table}):
        local={r["id"] for r in table if r["building"]==building}
        vv=v[np.isin(ids,list(local))];a,b=vv.min(0),vv.max(0)
        boxes.append(((a+b)/2,float(np.linalg.norm(b-a))*.58))
    close=max(boxes,key=lambda b:float(b[0]@view))
    block=((lo+hi)/2,float(np.linalg.norm(hi-lo))*.57)
    def render(camera,lane,day,prob,debug=0):
        center,scale=camera
        m=np.eye(4);m[:3,:3]=np.array([right/scale,up/scale,-view/dscale])
        m[:3,3]=-m[:3,:3]@center
        prog["mvp"].write(m.T.astype("f4").tobytes())
        prog["lane"].value=lane;prog["daylight"].value=day;prog["occupancy"].value=prob
        prog["debug_mode"].value=debug;prog["dynamic_night"].value=1
        fb.clear(-1.,-1.,-1.,-1.,depth=1.) if debug else fb.clear(.68 if day>.5 else .008,.72 if day>.5 else .013,.74 if day>.5 else .025,1.,depth=1.)
        vao.render(moderngl.TRIANGLES)
        a=np.frombuffer(fb.read(components=4,dtype="f4",alignment=1),dtype=np.float32).reshape(size,size,4).copy()
        return a,depth.read()
    def image(a):
        return Image.fromarray(np.uint8(np.round(srgb_encode(a[::-1,:,:3])*255)))
    meta,md=render(block,2,1.,.5,1);layer,_=render(block,2,1.,.5,2)
    valid=meta[...,0]>=0;assert valid.sum()>100
    rr=np.rint(meta[...,0][valid]).astype(int);assert rr.min()>=0 and rr.max()<len(table)
    expected_layers=np.array([LAYERS.index(r["family"]) for r in table])
    np.testing.assert_array_equal(np.rint(layer[...,1][valid]).astype(int),expected_layers[rr])
    np.testing.assert_array_equal(np.rint(layer[...,0][valid]).astype(int),rr)
    checked=0;ambiguous=0
    for i in np.unique(rr):
        r=table[int(i)];sel=valid&(np.rint(meta[...,0])==i)
        samples=meta[...,1:3][sel];actual=np.rint(meta[...,3][sel]).astype(int)
        if r["role"]!="wall":assert np.all(actual==0);continue
        want=lookup(samples,r,cells)[0];bad=want!=actual
        if np.any(bad):
            bb=samples[bad];aa=actual[bad];ok=np.zeros(len(bb),bool)
            for x in (-2e-4,0.,2e-4):
                for y in (-2e-4,0.,2e-4):
                    ok|=lookup(bb+[x,y],r,cells)[0]==aa
            assert ok.all(),"GLSL opening lookup differs away from boundary"
            ambiguous+=int(bad.sum())
        checked+=len(samples)
    low,_=render(block,2,.04,.10,3);high,_=render(block,2,.04,.68,3)
    roles=np.rint(meta[...,3]).astype(int)
    lit=np.any(high[...,:3]>1e-7,axis=-1)&valid
    assert np.all(np.isin(roles[lit],[1,2])) and np.all(high[valid,:3]+1e-6>=low[valid,:3])
    low_energy=float(low[valid,:3].sum());high_energy=float(high[valid,:3].sum())
    assert high_energy>low_energy
    sheet=Image.new("RGB",(size*3,size*2+32),(220,220,220))
    ImageDraw.Draw(sheet).text((8,8),"Actual GLSL/EGL software renderer / shared arrays / packed ABI / zero opening geometry",fill=(0,0,0))
    lanes=[(close,0,1.,0.,"close_flat"),(close,1,1.,0.,"close_pbr"),(close,2,1.,0.,"close_parallax"),
           (block,2,1.,0.,"block_day"),(block,2,.04,.68,"block_night_high"),(block,2,.04,.10,"block_night_low")]
    depths={}
    for i,(camera,lane,day,prob,name) in enumerate(lanes):
        a,d=render(camera,lane,day,prob);key=i//3
        if key in depths:assert depths[key]==d
        else:depths[key]=d
        assert np.isfinite(a).all()
        im=image(a);im.save(OUT/(name+".png"))
        ImageDraw.Draw(im).text((8,8),name,fill=(225,225,225) if day<.5 else (0,0,0))
        sheet.paste(im,((i%3)*size,(i//3)*size+32))
    sheet.save(OUT/"contact.png")
    report={"renderer":renderer,"gl_version":ctx.version_code,"actual_glsl_compile":True,
        "source_tile":chosen["manifest"],"buildings":chosen["buildings"],"triangles":len(faces),
        "one_indexed_draw_per_view":True,"opening_triangles_added":0,"material_layers":len(LAYERS),
        "shared_window_door_modules":6,"module_size":256,"base_source_size":manifest["resolution"],
        "visible_record_pixels":int(valid.sum()),"glsl_opening_lookup_samples":checked,
        "near_boundary_ambiguities":ambiguous,"material_layer_lookup_exact":True,
        "emission_glazing_only":True,"low_probability_energy":low_energy,"high_probability_energy":high_energy,
        "depth_identical_within_camera":True,"unique_building_image_textures":0,
        "limitations":["software OpenGL, not UUSL/UNIGINE or hardware GPU","synthetic source textures and approximate GGX/analytic reflection, no shadows/GI",
        "simple parallax shifts glass interior only","global illustrative occupancy probability, not full per-use profile ABI",
        "module mip filtering but no measured flight anti-aliasing or native LOD","no native GPU timing, draw count only in this lab"]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("GLSL_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1100,760));b=io.BytesIO();thumb.save(b,format="JPEG",quality=70,optimize=True)
    data=base64.b64encode(b.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"GLSL_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main()
