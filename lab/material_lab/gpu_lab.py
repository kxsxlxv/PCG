"""Independent GPU material lab on actual stage 21/22 export; no proprietary maps in Git."""
from pathlib import Path
import argparse,io,json,os,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
import scene_io
from loggia import fit,affine
from egl_backend import gl,gen,shader,ptr,C,I,U,P,F,B
HERE=Path(__file__).parent

def srgb(a):
    a=np.asarray(a,np.float32);return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
def rgba(path,size):return np.asarray(Image.open(path).convert('RGBA').resize((size,size),Image.Resampling.LANCZOS),np.float32)/255.
def rgb(path,size):return rgba(path,size)[:,:,:3]
def linear(a):
    a=a.copy();a[:,:,:3]=srgb(a[:,:,:3]);return a
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little')%4096

def material_arrays(materials,panelka,source,assets,private=None,hdr=None,size=512):
    A=[];N=[];R=[];used=[]
    def add(family,paths=None,rough=.7,metal=0.):
        d=materials/family;al=paths[0] if paths else d/'basecolor.png';normal=paths[1] if paths else d/'normal.png';orm=paths[2] if paths else d/'orm.png'
        a=linear(rgba(al,size));n=rgb(normal,size) if normal and normal.exists() else np.broadcast_to([.5,.5,1.],(size,size,3)).copy()
        if paths:
            r=np.ones((size,size,3),np.float32);r[:,:,1]=rgb(orm,size)[:,:,0] if orm and orm.exists() else rough;r[:,:,2]=metal
        elif orm.exists():r=rgb(orm,size)
        else:r=np.broadcast_to([1.,rough,metal],(size,size,3)).copy()
        A.append(a);N.append(n);R.append(r);used.append(str(al.name))
    def trip(a,b,c=None):return [panelka/a,panelka/b,panelka/c if c else None]
    add('red_brick',trip('Bricks2diffuse.png','Bricks2normal.png','Bricks2roughness.png'))
    for f in ('concrete_panel','concrete_panel','plaster_cream','stone_plinth','bitumen_roof'):add(f)
    add('ceramic_roof',trip('Slating1diffuse.png','Slating1normal.png'),.76,.08)
    for f in ('plaster_white','plaster_white','bitumen_roof','plaster_white'):add(f)
    add('concrete_panel',trip('GreenMetalDoordiffuse.png','GreenMetalDoornormal.png'),.56,.20)
    add('concrete_panel',trip('Went1diffuse.png','Went1normal.png'),.63,.18)
    for i,c in [(7,[.55,.53,.49]),(8,[.30,.32,.31]),(9,[.42,.45,.45])]:
        A[i][:,:,:3]=np.array(c)*(.92+.08*A[i][:,:,:3]);R[i][:,:,1]=.63 if i!=9 else .4;R[i][:,:,2]=.04 if i!=9 else .5
    masks=[]
    for name in ('roof_balcony_cutout.glb','antenna_cutout.glb'):
        path=source/name;raw=path.read_bytes();n=int.from_bytes(raw[12:16],'little');blob=raw[28+n:]
        for p,a,f,t in scene_io.glb(path):
            tex=t['materials'][p['material']]['pbrMetallicRoughness']['baseColorTexture']['index'];img=t['images'][t['textures'][tex]['source']];v=t['bufferViews'][img['bufferView']];off=v.get('byteOffset',0)
            im=Image.open(io.BytesIO(blob[off:off+v['byteLength']])).convert('RGBA').resize((1024,1024),Image.Resampling.NEAREST);masks.append(linear(np.asarray(im,np.float32)/255.))
    assert len(masks)==3
    room=[];room_source=0;rooms=assets/'three-fenestra'/'rooms.webp';indices=[0,2,4,5,9,11,14,15]
    if rooms.exists():
        image=Image.open(rooms).convert('RGB');w,h=image.size
        for i in indices:
            y,x=divmod(i,4);im=image.crop((round(x*w/4)+2,round(y*h/4)+2,round((x+1)*w/4)-2,round((y+1)*h/4)-2)).resize((512,512),Image.Resampling.LANCZOS);room.append(srgb(np.asarray(im,np.float32)/255.))
        room_source=1
    else:room=[np.zeros((512,512,3),np.float32) for i in indices]
    lobby={};private_used=[]
    if private:
        if os.getenv('GITHUB_ACTIONS'):raise RuntimeError('private EXR maps must not enter CI')
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1');import cv2
        for mode in ('day','night'):
            candidates=sorted(private.glob('*LB01*'+mode+'*2K.exr*'))
            if not candidates:continue
            p=candidates[0];a=cv2.imdecode(np.fromfile(p,np.uint8),cv2.IMREAD_UNCHANGED)
            if a is None or a.ndim!=3 or a.shape[2]<3:raise ValueError(p.name)
            a=np.maximum(a[:,:,[2,1,0]],0);assert np.isfinite(a).all();a=cv2.resize(a,(512,512),interpolation=cv2.INTER_AREA);lobby[mode]=len(room);room.append(a);private_used.append(p.name)
    if hdr:
        import cv2
        e=cv2.imread(str(hdr),cv2.IMREAD_UNCHANGED);assert e is not None;e=np.maximum(e[:,:,::-1],0);e=cv2.resize(e,(1024,512),interpolation=cv2.INTER_AREA);assert np.isfinite(e).all()
    else:
        theta,phi=np.meshgrid((np.arange(1024)+.5)/1024*2*np.pi-np.pi,(np.arange(512)+.5)/512*np.pi);z=np.cos(phi);x=np.cos(theta)*np.sin(phi);y=np.sin(theta)*np.sin(phi);sun=np.array([.5,-.7,.63]);sun/=np.linalg.norm(sun);dot=x*sun[0]+y*sun[1]+z*sun[2];sky=np.clip(z,0,1);e=np.stack([.50-.24*sky,.61-.17*sky,.74-.06*sky],axis=-1);e[z<0]=[.16,.19,.15];e+=np.exp((dot-1)*90)[:,:,None]*np.array([3.2,2.4,1.6])
    return [np.asarray(a,np.float32) for a in (A,N,R,masks,room,e)],dict(room_source=room_source,lobby=lobby,private_used=private_used,material_files=used,hdr=bool(hdr))

def build_metadata(scene,cells,source):
    v,u,ids,table=scene;model=fit(scene);by={d['record']['id']:m for m in model.values() for d in m['faces'].values()};pairs=json.loads((source/'opening_blocks_v2.json').read_text())['paired_openings'];rows=[];meta=np.zeros((len(table),8,4),np.float32)
    families={'red_brick':(0,1.2),'concrete_panel':(2,2.4),'plaster_cream':(3,2.),'stone_plinth':(4,1.),'bitumen_roof':(5,2.),'ceramic_roof':(6,1.6),'painted_trim':(7,1.),'painted_rail':(8,.8),'metal_roof_red':(6,1.6)}
    for r in table:
        j=r['id'];k=r.get('accessory','');mask=ids==j;layer,repeat=families.get(r.get('family'),(2,2.4));meta[j,0]=[0,layer,repeat,seed(r.get('surface',k))]
        if not mask.any():continue
        if k.startswith('cutout_'):
            meta[j,0,:2]=[3,('cutout_roof','cutout_balcony','cutout_antenna').index(k)];continue
        if k in ('coping','gutter_side','gutter_floor','bent_drain','roof24_cowl','roof24_drain','roof24_sill','stage23_drain'):meta[j,0,1:3]=[9,1.]
        if k=='vent':meta[j,0,1:3]=[12,1.]
        if k=='ac':
            w,h=r['span'];meta[j,0,0]=5;meta[j,1,:3]=[w,h,float(w>.45 and h>.40)];continue
        if j in by:
            m=by[j];host=m['host'];meta[j,0,0]=2;key=seed(r['placement']);meta[j,1,:3]=[host['id'],(.27,.34,.40)[key%3],(key%100)/100.];meta[j,2,:3]=m['origin'];meta[j,3,:3]=m['frame'][:,0];meta[j,4,:3]=m['frame'][:,1];meta[j,5,:3]=m['lo'];meta[j,6,:3]=m['hi'];meta[j,7,:2]=r['span'];continue
        if r.get('role')!='wall':continue
        style=0 if r['building'].endswith('panel_block') else 1 if r['building'].endswith('brick_stack') else 2
        layer,repeat=[(1,2.4),(0,1.2),(3,2.)][style];meta[j,0,:3]=[1,layer,repeat];bays=r['bays'];floors=r['floors'];start=len(rows);meta[j,1]=[start,bays,floors,style];meta[j,2,:3]=[r['first_center'],r['spacing'],r['floor_h']];a=affine(v[mask],u[mask]);t=a[0]/np.linalg.norm(a[0]);n=np.cross(t,a[1]);n/=np.linalg.norm(n);meta[j,4,:3]=t;meta[j,5,:3]=n
        for slot in range(bays*floors):
            q=np.zeros((3,4),np.float32);code=int(cells[r['cell_start']+slot])&3;f,b=divmod(slot,bays);key=seed(r['facade_id']+':'+str(slot));pair=pairs.get(str(j),{}).get(str(slot));center=r['first_center']+b*r['spacing'];w=r['window_width']
            if pair:q[0]=pair['door'];q[1]=pair['window'];q[2,:3]=[2,1,key]
            elif code:
                z=f*r['floor_h']+(r['balcony_threshold'] if code==2 else r['sill']);h=r['door_height'] if code==2 else r['window_height'];q[0]=[center-w/2,z,center+w/2,z+h];q[2,:3]=[code,0,key]
            rows.append(q)
        q=np.zeros((3,4),np.float32)
        if r['door'] is not None:
            x,y,w,h=r['door'];q[0]=[x,y,x+w,y+h];q[2,:3]=[3 if style==2 else 4,0,seed(r['facade_id']+':entry')]
        rows.append(q)
    return meta,np.asarray(rows,np.float32),model

def vertex_data(scene):
    v,u,ids,_=scene;assert len(v)%3==0;P=v.reshape(-1,3,3);T=u.reshape(-1,3,2);e1=P[:,1]-P[:,0];e2=P[:,2]-P[:,0];n=np.cross(e1,e2);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12);du=T[:,1]-T[:,0];dv=T[:,2]-T[:,0];det=du[:,0]*dv[:,1]-du[:,1]*dv[:,0];tt=e1*dv[:,1,None]-e2*du[:,1,None];tt*=np.where(det<0,-1.,1.)[:,None];small=np.linalg.norm(tt,axis=1)<1e-8;tt[small]=e1[small];tt/=np.maximum(np.linalg.norm(tt,axis=1,keepdims=True),1e-12)
    return np.ascontiguousarray(np.column_stack((v,u,ids,np.repeat(n,3,axis=0),np.repeat(tt,3,axis=0))),np.float32)

def norm(x):x=np.asarray(x,float);return x/np.linalg.norm(x)
def camera(center,view,W,H,ppm,perspective=False):
    d=norm(view);right=norm(np.cross([0,0,1],d));up=np.cross(d,right);center=np.asarray(center,float)
    if perspective:
        dist=H/ppm/(2*np.tan(np.deg2rad(22.5)));eye=center+d*dist;V=np.eye(4);V[:3,:3]=[right,up,d];V[:3,3]=-V[:3,:3]@eye;near=.12;far=300.;f=1/np.tan(np.deg2rad(22.5));P=np.array([[f*H/W,0,0,0],[0,f,0,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]])
    else:
        eye=center+d*200;V=np.eye(4);V[:3,:3]=[right,up,d];V[:3,3]=-V[:3,:3]@center;P=np.diag([2*ppm/W,2*ppm/H,-1/200,1])
    return np.asarray(P@V,np.float32),eye,d
VERT='''#version 330 core
layout(location=0) in vec3 p;layout(location=1) in vec2 st;layout(location=2) in float record;layout(location=3) in vec3 normal;layout(location=4) in vec3 tan;
uniform mat4 VP;out vec3 world;out vec2 uv;flat out int rid;flat out vec3 gn;flat out vec3 tangent;
void main(){world=p;uv=st;rid=int(record+.5);gn=normal;tangent=tan;gl_Position=VP*vec4(p,1);}
'''
class GL:
    def __init__(self,vertices,meta,ops,arrays):
        self.p=gl('glCreateProgram',U)();frag='#version 330 core\n'+(HERE/'material_math.glsl').read_text()+(HERE/'renderer.glsl').read_text()
        for typ,s in ((0x8B31,VERT),(0x8B30,frag)):gl('glAttachShader',None,U,U)(self.p,shader(typ,s))
        gl('glLinkProgram',None,U)(self.p);v=I();gl('glGetProgramiv',None,U,U,P)(self.p,0x8B82,C.byref(v));assert v.value;gl('glUseProgram',None,U)(self.p)
        self.vao=gen('glGenVertexArrays');gl('glBindVertexArray',None,U)(self.vao);self.vbo=gen('glGenBuffers');gl('glBindBuffer',None,U,U)(0x8892,self.vbo);gl('glBufferData',None,U,C.c_ssize_t,P,U)(0x8892,vertices.nbytes,ptr(vertices),0x88E4)
        for i,n,offset in ((0,3,0),(1,2,3),(2,1,5),(3,3,6),(4,3,9)):
            gl('glEnableVertexAttribArray',None,U)(i);gl('glVertexAttribPointer',None,U,I,U,C.c_ubyte,I,P)(i,n,0x1406,0,48,P(offset*4))
        self.count=len(vertices);gl('glEnable',None,U)(0x0B71);gl('glDisable',None,U)(0x0B44);gl('glDisable',None,U)(0x0BE2)
        a,n,r,m,room,e=arrays;self.tex2(meta,0,'metadata',False);self.tex2(ops,1,'openings',False);self.texarray(a,2,'albedoMaps');self.texarray(n,3,'normalMaps');self.texarray(r,4,'ormMaps');self.texarray(room,5,'roomMaps');self.tex2(e,6,'envMap',True,True);self.texarray(m,8,'cutoutMaps');self.shadow=2048;self.sfb,self.sd,_=self.framebuffer(self.shadow,self.shadow,True);self.unit(7,0x0DE1,self.sd);self.uniform('shadowMap',7);self.fbos={}
    def uniform(self,name,x):
        loc=gl('glGetUniformLocation',I,U,B)(self.p,name.encode());
        if loc<0:return
        if isinstance(x,(int,np.integer)):gl('glUniform1i',None,I,I)(loc,int(x))
        elif np.isscalar(x):gl('glUniform1f',None,I,F)(loc,float(x))
        else:
            a=np.ascontiguousarray(x,np.float32)
            if a.shape==(4,4):gl('glUniformMatrix4fv',None,I,I,C.c_ubyte,P)(loc,1,1,ptr(a))
            elif a.shape==(3,):gl('glUniform3fv',None,I,I,P)(loc,1,ptr(a))
            else:raise ValueError(name)
    def unit(self,u,target,tex):gl('glActiveTexture',None,U)(0x84C0+u);gl('glBindTexture',None,U,U)(target,tex)
    def param(self,t,k,v):gl('glTexParameteri',None,U,U,I)(t,k,v)
    def tex2(self,a,unit,name,mip,repeat=False):
        a=np.ascontiguousarray(a,np.float32);h,w,c=a.shape;t=gen('glGenTextures');self.unit(unit,0x0DE1,t);gl('glTexImage2D',None,U,I,I,I,I,I,U,U,P)(0x0DE1,0,0x8814 if c==4 else 0x8815,w,h,0,0x1908 if c==4 else 0x1907,0x1406,ptr(a));self.param(0x0DE1,0x2801,0x2703 if mip else 0x2600);self.param(0x0DE1,0x2800,0x2601 if mip else 0x2600);self.param(0x0DE1,0x2802,0x2901 if repeat else 0x812F);self.param(0x0DE1,0x2803,0x812F)
        if mip:gl('glGenerateMipmap',None,U)(0x0DE1)
        self.uniform(name,unit);return t
    def texarray(self,a,unit,name):
        a=np.ascontiguousarray(a,np.float32);l,h,w,c=a.shape;t=gen('glGenTextures');self.unit(unit,0x8C1A,t);gl('glTexImage3D',None,U,I,I,I,I,I,I,U,U,P)(0x8C1A,0,0x881A if c==4 else 0x881B,w,h,l,0,0x1908 if c==4 else 0x1907,0x1406,ptr(a));self.param(0x8C1A,0x2801,0x2703);self.param(0x8C1A,0x2800,0x2601);self.param(0x8C1A,0x2802,0x2901 if name not in ('roomMaps','cutoutMaps') else 0x812F);self.param(0x8C1A,0x2803,0x2901 if name not in ('roomMaps','cutoutMaps') else 0x812F);gl('glGenerateMipmap',None,U)(0x8C1A)
        maxa=F();gl('glGetFloatv',None,U,P)(0x84FF,C.byref(maxa));
        if maxa.value>=1:gl('glTexParameterf',None,U,U,F)(0x8C1A,0x84FE,min(8.,maxa.value))
        self.uniform(name,unit);return t
    def framebuffer(self,w,h,depthonly=False):
        fb=gen('glGenFramebuffers');gl('glBindFramebuffer',None,U,U)(0x8D40,fb);d=gen('glGenTextures');self.unit(10,0x0DE1,d);gl('glTexImage2D',None,U,I,I,I,I,I,U,U,P)(0x0DE1,0,0x8CAC,w,h,0,0x1902,0x1406,None);self.param(0x0DE1,0x2801,0x2600);self.param(0x0DE1,0x2800,0x2600);self.param(0x0DE1,0x2802,0x812F);self.param(0x0DE1,0x2803,0x812F);gl('glFramebufferTexture2D',None,U,U,U,U,I)(0x8D40,0x8D00,0x0DE1,d,0);c=None
        if depthonly:gl('glDrawBuffer',None,U)(0);gl('glReadBuffer',None,U)(0)
        else:
            c=gen('glGenTextures');self.unit(10,0x0DE1,c);gl('glTexImage2D',None,U,I,I,I,I,I,U,U,P)(0x0DE1,0,0x881A,w,h,0,0x1908,0x1406,None);gl('glFramebufferTexture2D',None,U,U,U,U,I)(0x8D40,0x8CE0,0x0DE1,c,0)
        assert gl('glCheckFramebufferStatus',U,U)(0x8D40)==0x8CD5;return fb,d,c
    def draw(self,fb,w,h):
        gl('glBindFramebuffer',None,U,U)(0x8D40,fb);gl('glViewport',None,I,I,I,I)(0,0,w,h);gl('glClearColor',None,F,F,F,F)(.50,.59,.70,0.);gl('glClear',None,U)(0x4000|0x100);gl('glDrawArrays',None,U,I,I)(4,0,self.count)
    def render(self,center,ppm,view,mode,res,info,perspective=False):
        w,h=res;sun=norm([.5,-.7,.63]);night=mode=='night';suncol=np.array([3.,2.65,2.2]) if not night else np.array([.055,.065,.09]);ambient=np.array([.44,.49,.55]) if not night else np.array([.045,.055,.075]);sh,_,_=camera([61,6,11],sun,self.shadow,self.shadow,self.shadow/160.)
        for k,x in dict(VP=sh,shadowVP=sh,viewDir=sun,eye=[0,0,0],perspective=0.,sunDir=sun,sunColor=suncol,irradiance=ambient,night=float(night),pixelsPerMeter=float(ppm),roomSource=info['room_source'],lobbyLayer=info['lobby'].get(mode,-1),passShadow=1).items():self.uniform(k,x)
        self.unit(7,0x0DE1,0);self.draw(self.sfb,self.shadow,self.shadow);self.unit(7,0x0DE1,self.sd)
        vp,eye,d=camera(center,view,w,h,ppm,perspective);self.uniform('VP',vp);self.uniform('eye',eye);self.uniform('viewDir',d);self.uniform('perspective',float(perspective));self.uniform('passShadow',0)
        if (w,h) not in self.fbos:self.fbos[(w,h)]=self.framebuffer(w,h)
        self.draw(self.fbos[(w,h)][0],w,h);a=np.empty((h,w,4),np.float32);gl('glReadPixels',None,I,I,I,I,U,U,P)(0,0,w,h,0x1908,0x1406,ptr(a));err=gl('glGetError',U)();assert err==0,hex(err);assert np.isfinite(a).all();a=a[::-1].copy();mask=a[:,:,3]>.5;assert mask.any();c=a[:,:,:3];c[~mask]=[.012,.020,.041] if night else [.50,.59,.70];c=np.maximum(c,0);x=c*(1.3 if night else 1.05);x=np.clip((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0,1);x=np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055);im=Image.fromarray(np.uint8(x*255+.5));return im,dict(foreground_pixels=int(mask.sum()),hdr_max=float(c[mask].max()),hdr_mean=float(c[mask].mean()))

def run(source,materials,panelka,assets,out,private=None,hdr=None,quick=False):
    out.mkdir(parents=True,exist_ok=True);scene_io.P=source;scene,cells=scene_io.load_scene();meta,ops,models=build_metadata(scene,cells,source);assert len(models)==55 and len(scene[0])//3==4516
    arrays,info=material_arrays(materials,panelka,source,assets,private,hdr);renderer=GL(vertex_data(scene),meta,ops,arrays);info['renderer']=gl('glGetString',B,U)(0x1F01).decode();del arrays
    cams=[('panel_full',(58,6,10.6),16*1080/640,(.35,-1,.22),'day',(1920,1080),False),('brick_full',(107,6,8.5),17*1080/640,(.32,-1,.22),'day',(1920,1080),False),('panel_detail',(50.4,0,10.5),88*1280/640,(.32,-1,.13),'day',(1280,1280),False),('panel_oblique',(50.4,0,10.5),88*1280/640,(.85,-1,.13),'day',(1280,1280),False),('night',(58,6,10.6),16*1080/640,(.35,-1,.22),'night',(1920,1080),False),('entrance',(15,0,1.35),95*1280/640,(.28,-1,.10),'day',(1280,1280),False),('entrance_night',(15,0,1.35),95*1280/640,(.28,-1,.10),'night',(1280,1280),False),('aerial',(61,6,10),8*1080/640,(.3,-1,1.65),'day',(1920,1080),False),('perspective',(58,4,10.6),28,(.45,-1,.19),'day',(1920,1080),True),('detail_4k',(50.4,0,10.5),88*2160/640,(.32,-1,.13),'day',(3840,2160),False)]
    if quick:cams=[cams[2],cams[5],cams[0]]
    sheet=Image.new('RGB',(1400,370*len(cams)),(226,228,228));draw=ImageDraw.Draw(sheet);rows=[]
    for i,(name,center,ppm,view,mode,res,persp) in enumerate(cams):
        st=time.monotonic();im,metrics=renderer.render(center,ppm,view,mode,res,info,persp);im.save(out/f'{name}_gpu.png');row=dict(name=name,resolution=list(res),perspective=persp,**metrics);rows.append(row);print(name,round(time.monotonic()-st,2),row,flush=True)
        old=source/f'{name}_candidate.png'
        for j,im0 in enumerate((Image.open(old) if old.exists() else None,im)):
            if im0 is not None:im0=im0.copy();im0.thumbnail((700,340));sheet.paste(im0,(j*700+(700-im0.width)//2,i*370+22))
            draw.text((j*700+8,i*370+5),name+(' | previous CPU' if j==0 else ' | GPU'),fill='black')
    sheet.save(out/'comparison.png');report=dict(schema='pcg.research.gpu-material-lab.v1',triangles=len(scene[0])//3,extra_triangles=0,glazed_modules=len(models),window_profiles=16,material_sources=info,views=rows,limitations=['Independent research renderer; not native UNIGINE or GPU hardware benchmark.','GGX direct lighting; approximate IBL with environment mips and constant diffuse irradiance, no full convolution or BRDF LUT.','Procedural analytic frames replace all low-resolution whole-window photos in this lane; Gaxil maps not yet incorporated into GPU profiles.','Loggia second-side glass is an environment approximation, no real multilayer transmission or furniture.','POM, atlas ray mapping and alpha discard run on GPU; private lobby uses rear/side/floor/ceiling cross faces only, not layered EXR foreground.','Mip filtering and anisotropy enabled; alpha coverage preservation and MSAA still pending in this renderer.','Roof family color corrected but ridge flashing/weathering and detailed facade seam/plinth material grammar need parity with CPU.','HDRI and private lobby optional local only; open CI uses procedural HDR sky and Three-Fenestra.'])
    (out/'report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('artifacts/zero-window-entrance-stage'));p.add_argument('--materials',type=Path,default=Path('artifacts/materials'));p.add_argument('--panelka',type=Path,default=Path('assets/third_party/panelka'));p.add_argument('--assets',type=Path,default=Path('assets/third_party/window_research'));p.add_argument('--out',type=Path,default=Path('artifacts/gpu-material-lab'));p.add_argument('--private',type=Path);p.add_argument('--hdr',type=Path);p.add_argument('--quick',action='store_true');a=p.parse_args();run(**vars(a))
