"""CPU reference for metric POM and three atlas families. No commercial
textures in Git. Three-Fenestra perspective atlas re-projection and Gaxil
alpha/RMW conventions are based on their pinned MIT source (see notices).
JIWindowBox uses a 3x3 cross, side/floor/ceiling/back plus two alpha layers.
"""
from pathlib import Path
import os, math
import numpy as np
from PIL import Image

CROPS={'Panel1.png':(125,94,380,368),'Panel2.png':(117,101,386,395),'Panel4.png':(119,78,374,364),'Panel5.png':(119,76,356,338),'Panel6.png':(122,82,366,347)}
ROOMS=np.array([0,2,4,5,9,11,14,15]);OVERLAYS=np.array([0,2,3,4,5,6,9,10,12,14,15])

def srgb(a):
    a=np.clip(a,0,1);return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
def rgb_image(path):return np.asarray(Image.open(path).convert('RGB'),np.float32)/255.
def rgba_image(path):return np.asarray(Image.open(path).convert('RGBA'),np.float32)/255.
def linear_rgba(a):
    b=a.copy();b[:,:,:3]=srgb(b[:,:,:3]);return b

def bilinear(a,uv,repeat=False):
    H,W=a.shape[:2];u,v=np.asarray(uv).T
    x=u*W-.5;y=v*H-.5
    if not repeat:x=np.clip(x,0,W-1);y=np.clip(y,0,H-1)
    i=np.floor(x).astype(np.int64);j=np.floor(y).astype(np.int64);s=(x-i)[:,None];t=(y-j)[:,None]
    I=(i+1)%W if repeat else np.minimum(i+1,W-1);J=(j+1)%H if repeat else np.minimum(j+1,H-1)
    i%=W;j%=H
    return (a[j,i]*(1-s)+a[j,I]*s)*(1-t)+(a[J,i]*(1-s)+a[J,I]*s)*t

class Mips:
    def __init__(self,a,normal=False):
        a=np.asarray(a,np.float32);self.levels=[a];self.normal=normal
        # Crop atlas cells before filtering; linear RGB or linear data.
        while min(a.shape[:2])>2:
            h,w=a.shape[:2];nw,nh=max(1,w//2),max(1,h//2)
            b=np.stack([np.asarray(Image.fromarray(a[:,:,i],mode='F').resize((nw,nh),Image.Resampling.BOX)) for i in range(a.shape[2])],axis=-1)
            if normal:
                q=b*2-1;q/=np.maximum(np.linalg.norm(q,axis=2,keepdims=True),1e-6);b=(q+1)/2
            self.levels.append(b);a=b
    def sample(self,uv,pixels,repeat=False):
        # Estimate footprint using actual projected opening pixels. Trilinear
        # between levels, not hardware anisotropy or per-fragment derivatives.
        level=float(np.clip(np.log2(max(self.levels[0].shape[:2])/max(float(pixels),1)),0,len(self.levels)-1));i=int(level);f=level-i
        a=bilinear(self.levels[i],uv,repeat)
        return a if not f or i==len(self.levels)-1 else a*(1-f)+bilinear(self.levels[i+1],uv,repeat)*f

def atlas_cells(a,rows,cols,normal=False):
    H,W=a.shape[:2];out=[]
    for i in range(rows):
        for j in range(cols):
            # Keep two-pixel inset to avoid adjacent-cell bleed at source edges.
            q=a[round(i*H/rows)+2:round((i+1)*H/rows)-2,round(j*W/cols)+2:round((j+1)*W/cols)-2]
            out.append(Mips(q,normal))
    return out

def load_exr(path):
    # OpenCV supports ordinary R,G,B,A EXRs. Do not silently flatten named
    # multichannel EXRs; the caller will record those as unsupported.
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    import cv2
    raw=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim!=3 or raw.shape[2]!=4:raise ValueError(f'not a standard RGBA EXR: {Path(path).name}')
    a=raw[:,:,[2,1,0,3]].astype(np.float32)
    if not np.isfinite(a).all():raise ValueError('nonfinite EXR')
    a[:,:,:3]=np.maximum(a[:,:,:3],0);a[:,:,3]=np.clip(a[:,:,3],0,1)
    return a

def cross_cells(a):
    # Keep tile order top-left to bottom-right. Five room faces are
    # (1,0) ceiling, (0,1) left, (1,1) rear, (2,1) right, (1,2) floor.
    return atlas_cells(a,3,3)

def hit_box(uv,ratio,depth,w,h):
    """uv is [0,1] with up-positive y; ratio is camera tangent / normal.
    Ray into room is (-ratio.x,-ratio.y,+1). Return local hit and face ID.
    Depth is metric and width/height can differ. Axial rays robust at zero.
    """
    uv=np.asarray(uv,float);d=-np.asarray(ratio,float)/np.array([w,h]);p=np.clip(uv,0,1)
    tx=np.full(len(p),np.inf);ty=tx.copy()
    if abs(d[0])>1e-10:tx=((1 if d[0]>0 else 0)-p[:,0])/d[0]
    if abs(d[1])>1e-10:ty=((1 if d[1]>0 else 0)-p[:,1])/d[1]
    T=np.column_stack((tx,ty,np.full(len(p),depth)));f=T.argmin(1);t=T[np.arange(len(p)),f]
    q=np.clip(p+t[:,None]*d,0,1);face=np.full(len(p),4,np.int8);face[f==0]=5 if d[0]>0 else 3;face[f==1]=1 if d[1]>0 else 7
    return q,t,face

def perspective_uv(uv,ratio,depth,w,h,backscale=.66):
    q,t,_=hit_box(uv,ratio,depth,w,h);dist=backscale*depth/(1-backscale)
    p=(q-.5)*(dist/(dist+t))[:,None]+.5
    return np.column_stack((p[:,0],1-p[:,1]))

def windowbox(cells,uv,ratio,depth,w,h,pixels,mid=True,front=True):
    # Independent ray-box implementation using the same atlas contract as
    # the Apache-2.0 JIWindowBox reference, including its layered alpha planes.
    q,t,face=hit_box(uv,ratio,depth,w,h);out=np.zeros((len(uv),3),float)
    for f in (1,3,4,5,7):
        ii=face==f
        if not ii.any():continue
        x,y=q[ii].T;k=t[ii]/depth
        if f==3:xy=np.c_[k,1-y]  # left: near edge outer, rear inner
        elif f==5:xy=np.c_[1-k,1-y]
        elif f==1:xy=np.c_[x,k]
        elif f==7:xy=np.c_[x,1-k]
        else:xy=np.c_[x,1-y]
        # Cross image orientation checked against the actual 2K atlas.
        out[ii]=cells[f].sample(xy,pixels)[:,:3]
    if mid:
        # wParallax ADV adds a second furniture and a people plane in
        # the right-hand corner cells. Composite back-to-front. Depth
        # fractions are calibrated defaults, not supplier scene metadata.
        for tile,fraction in ((0,.64),(2,.40),(8,.25)):
            plane=fraction*depth;p=uv-np.asarray(ratio)*plane/np.array([w,h]);ok=(t>=plane)&np.all((p>=0)&(p<=1),axis=1)
            if ok.any():
                a=cells[tile].sample(np.c_[p[ok,0],1-p[ok,1]],pixels);alpha=a[:,3:4];out[ok]=out[ok]*(1-alpha)+a[:,:3]*alpha
    if front:
        a=cells[6].sample(np.c_[uv[:,0],1-uv[:,1]],pixels);alpha=a[:,3:4];out=out*(1-alpha)+a[:,:3]*alpha
    return out

def profile(q,w,h,layout='double',door=False,photographic=False):
    # Depth measured INWARD from facade plane; no height-from-luminance.
    x,y=q.T;edge=np.minimum(np.minimum(x,w-x),np.minimum(y,h-y));outside=edge<0
    d=np.full(len(q),.125);kind=np.full(len(q),3,np.uint8) # glass
    reveal=edge<.052;frame=edge<.094
    kind[frame]=2;d[frame]=.065
    kind[reveal]=1;d[reveal]=.025+np.clip(edge[reveal],0,.052)*.65
    if not photographic:
        if layout=='double':m=np.abs(x-w*.5)<.027
        elif layout=='asym':m=np.abs(x-w*.38)<.027
        elif layout=='triple':m=np.minimum(np.abs(x-w/3),np.abs(x-2*w/3))<.025
        elif layout=='transom':m=(np.abs(x-w*.5)<.027)|(np.abs(y-h*.72)<.026)
        else:m=np.zeros(len(q),bool)
        if door:m=np.abs(y-h*.30)<.027
        m&=~frame;d[m]=.073;kind[m]=2
    sill=(y>=0)&(y<.036)&~outside;d[sill]=.016;kind[sill]=4
    if door:
        panel=(y<h*.30)&(kind==3);d[panel]=.09;kind[panel]=5
    d[outside]=0;kind[outside]=1
    return d,kind

def pom(uv,ratio,w,h,layout='double',door=False,photographic=False,steps=16):
    q=np.asarray(uv,float)*[w,h];ratio=np.asarray(ratio,float)
    if steps<=0:
        depth,kind=profile(q,w,h,layout,door,photographic);return q/[w,h],kind,depth
    # March from wall plane to glass, then refine crossing within the layer.
    # Treat a ray leaving the aperture as hitting the opaque reveal rather
    # than wrapping to the opposite edge or reading a neighboring room.
    delta=.14/steps;last=np.zeros(len(q));done=np.zeros(len(q),bool);depth=np.full(len(q),.14);hitkind=np.full(len(q),3,np.uint8)
    prev=profile(q,w,h,layout,door,photographic)[0]
    for i in range(1,steps+1):
        z=i*delta;p=q-ratio*z;D,K=profile(p,w,h,layout,door,photographic)
        new=(~done)&(z>=D)
        if new.any():
            a=prev[new]-(z-delta);b=z-D[new];f=np.clip(a/np.maximum(a+b,1e-8),0,1)
            depth[new]=z-delta+delta*f;hitkind[new]=K[new];done[new]=True
        prev=D
        if done.all():break
    p=q-ratio*depth[:,None];D,kind=profile(p,w,h,layout,door,photographic)
    return np.clip(p/[w,h],0,1),hitkind,depth

class Library:
    def __init__(self,root,private=None):
        root=Path(root);P=root/'panelka';W=root/'window_research';G=W/'gaxil';T=W/'three-fenestra'
        self.photos=[Mips(srgb(np.asarray(Image.open(P/k).convert('RGB').crop(v),np.float32)/255)) for k,v in CROPS.items()]
        self.brick=[Mips(srgb(rgb_image(P/f'Bricks{i}diffuse.png'))) for i in (1,2)]
        self.brick_normal=[Mips(rgb_image(P/f'Bricks{i}normal.png'),True) for i in (1,2)]
        self.brick_rough=[Mips(rgb_image(P/k)) for k in ('Brick1roughness.png','Bricks2roughness.png')]
        self.concrete=Mips(srgb(np.asarray(Image.open(P/'Panel1Edge.png').convert('RGB').crop((80,45,540,445)),np.float32)/255))
        self.door=Mips(srgb(rgb_image(P/'GreenMetalDoordiffuse.png')));self.door_normal=Mips(rgb_image(P/'GreenMetalDoornormal.png'),True)
        self.slate=Mips(srgb(rgb_image(P/'Slating1diffuse.png')));self.slate_normal=Mips(rgb_image(P/'Slating1normal.png'),True)
        self.vent=Mips(srgb(rgb_image(P/'Went1diffuse.png')));self.vent_normal=Mips(rgb_image(P/'Went1normal.png'),True)
        # Gaxil: four blind-state tiles stacked vertically. Crop individual
        # tall panes, not all four panes compressed into one opening.
        A=linear_rgba(rgba_image(G/'BuildingTexture_basecolorFull.png'));R=rgb_image(G/'BuildingTexture_RMW.png');N=rgb_image(G/'BuildingTexture_normal3.png')
        self.gaxil=[];self.gaxil_rmw=[];self.gaxil_normal=[]
        for j in range(4):
            x0=(16+122*j)/512;x1=(130+122*j)/512
            def crop(a,y0,y1):H,W=a.shape[:2];return a[round(y0*H):round(y1*H),round(x0*W):round(x1*W)]
            self.gaxil_rmw.append(Mips(crop(R,81/512,328/512)));self.gaxil_normal.append(Mips(crop(N,81/512,328/512),True))
            self.gaxil.append([Mips(crop(A,(i+81/512)/4,(i+328/512)/4)) for i in range(4)])
        self.rooms=atlas_cells(srgb(rgb_image(T/'rooms.webp')),4,4);self.overlay=atlas_cells(linear_rgba(rgba_image(T/'overlay.webp')),4,4)
        self.private={};self.errors=[]
        if private:
            for path in sorted(Path(private).glob('*.exr*')):
                try:self.private[path.name]=cross_cells(load_exr(path))
                except ValueError as e:self.errors.append(str(e))
    def lobby(self,uv,ratio,w,h,pixels,mode,key):
        maps=sorted(k for k in self.private if '_'+mode+'_' in k and 'Lobby' in k)
        if maps:
            # Size family follows opening width, not random apartment selection.
            size='S' if w<2.1 else 'M' if w<4.5 else 'L'
            valid=[k for k in maps if '_'+size+'_' in k] or maps
            return windowbox(self.private[valid[key%len(valid)]],uv,ratio,3.8,w,h,pixels), 'JIWindowBox/lobby'
        # Open CI fallback: corridor image, not a claim that paid EXRs ran.
        return self.rooms[15].sample(perspective_uv(uv,ratio,2.4,w,h),pixels), 'Three-Fenestra/corridor'

def material_response(rgb,normal,rough,ratio,metal=0.):
    # CPU approximation of normal/roughness response, not a physical IBL BRDF.
    n=normal*2-1;n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-6)
    light=np.array([.32,.48,.82]);light/=np.linalg.norm(light)
    v=np.r_[ratio,1.];v/=np.linalg.norm(v);half=light+v;half/=np.linalg.norm(half)
    ndl=np.clip(n@light,0,1);ndh=np.clip(n@half,0,1);rough=np.clip(rough,.08,1)
    spec=(.025+.10*metal)*(ndh**(2+28*(1-rough)))
    return rgb*(.76+.30*ndl)[:,None]+spec[:,None]

class WindowShader:
    def __init__(self,base,lib,cells,pairs,scene,roof_info,roof_frames):
        from zero_window_blocks import openings,seed
        self.base=base;self.lib=lib;self.pairs=pairs;self.info=roof_info;self.roof_frames=roof_frames;self.seed=seed;self.rects={};self.axes={};self.lanes={};self.counts={}
        v,uv,ids,table=scene
        for r in table:
            if r.get('role')=='wall' and 'accessory' not in r:self.rects[r['id']]=openings(r,cells)
            if r['id'] not in self.rects and r.get('accessory')!='stage21_glazed':continue
            p=v[ids==r['id']];q=uv[ids==r['id']]
            if len(p)<3:continue
            a=np.linalg.lstsq(np.c_[q,np.ones(len(q))],p,rcond=None)[0]
            t=a[0]/np.linalg.norm(a[0]);b=a[1]/np.linalg.norm(a[1]);n=np.cross(t,b);n/=np.linalg.norm(n)
            self.axes[r['id']]=(t,b,n)
        self.configure((.3,-1,.15),32,1080)
    def configure(self,view,ppm,height):
        self.view=np.asarray(view,float);self.view/=np.linalg.norm(self.view);self.ppm=ppm*height/640
    def angle(self,r,w,h):
        t,b,n=self.axes[r['id']];dot=float(self.view@n);d=max(.14,abs(dot));ratio=np.clip(np.array([self.view@t,self.view@b])/d,-5,5)
        pix=self.ppm*max(w,h)*max(.23,d);return ratio,pix,d
    def room(self,uv,ratio,w,h,pix,key):
        # Avoid repeating luxury bathrooms, gym and storage on ordinary homes.
        idx=int(ROOMS[key%len(ROOMS)]);q=perspective_uv(uv,ratio if pix>12 else np.zeros(2),2.6,w,h)
        return self.lib.rooms[idx].sample(q,pix)
    def glass(self,room,ndv,mode,lit):
        fres=.045+.55*(1-np.clip(ndv,0,1))**5
        sky=srgb(np.array([.46,.57,.65]));c=room*.58*(1-fres)+sky*fres
        if mode=='night':
            c=room*.17+sky*fres*.30
            e=room*(.78 if lit else .012)*np.array([1.,.86,.68])
        else:e=np.zeros_like(c)
        return c,e
    def opening(self,s,r,rect,key,mode,kind,shared=None,force=None):
        x,y,X,Y=rect;w=X-x;h=Y-y;uv=(s-[x,y])/[w,h];ratio,pix,ndv=self.angle(r,w,h)
        lit=(key%100)<36;door=kind==2;entry=kind==3;style=r['building'].split(':')[-1]
        if entry and style!='mixed_use':
            q=np.c_[uv[:,0],1-uv[:,1]];rgb=self.lib.door.sample(q,pix);N=self.lib.door_normal.sample(q,pix)
            return material_response(rgb,N,np.full(len(q),.58),ratio),np.zeros_like(rgb),'Panelka/metal_door'
        lane=force or ('lobby' if entry else 'paired' if shared else ('photo','three','photo','gaxil','three')[key%5])
        if lane=='photo' and r['building'].endswith('mixed_use') and int(y/r['floor_h'])==0:lane='commercial'
        # Opaque photo/atlas frame profiles already include mullions. Use
        # metric outer reveal only; no duplicate procedural interior bars.
        layout=('double','asym','triple','transom')[key%4]
        authored=lane in ('photo','three','gaxil','commercial')
        steps=20 if pix>=55 else 8 if pix>=20 else 0
        p,mask,depth=pom(uv,ratio,w,h,layout,door,authored,steps)
        rgb=np.empty((len(s),3),float);em=np.zeros_like(rgb)
        frame=srgb(np.array([.76,.75,.70]) if key%6 else np.array([.45,.32,.23]));rgb[:]=frame
        rgb[mask==1]=srgb(np.array([.43,.42,.39]))*(.65+.35*np.clip(depth[mask==1]/.06,0,1))[:,None]
        rgb[mask==4]=srgb(np.array([.62,.65,.63]));rgb[mask==5]=frame*.62
        pane=mask==3
        if not pane.any():return rgb,em,lane
        q=p[pane];imageuv=np.c_[q[:,0],1-q[:,1]]
        if lane=='photo':
            # Ground-floor barred photo is not scattered through upper floors.
            idx=[0,1,3,4][key%4] if y>=r['floor_h'] else key%5
            photo=self.lib.photos[idx].sample(imageuv,pix)
            rgb[pane]=photo
            if mode=='night' and lit:em[pane]=photo*.30*np.array([1.,.80,.52])
            return rgb,em,'Panelka/photo'
        if shared:
            a,b,A,B=shared;roomuv=(s[pane]-[a,b])/[A-a,B-b];room=self.room(np.clip(roomuv,0,1),ratio,A-a,B-b,pix,key)
        elif lane=='lobby':room,lane=self.lib.lobby(q,ratio,w,h,pix,mode,key)
        elif lane=='commercial':
            maps=sorted(k for k in self.lib.private if 'Free_Scene2' in k)
            if maps:room=windowbox(self.lib.private[maps[key%len(maps)]],q,ratio,3.8,w,h,pix);lane='JIWindowBox/free-commercial'
            else:room=self.room(q,ratio,w,h,pix,key);lane='Three-Fenestra/commercial-fallback'
        else:room=self.room(q,ratio,w,h,pix,key)
        c,e=self.glass(room,ndv,mode,lit or entry)
        if lane=='gaxil':
            # Actual alpha=opacity in Gaxil source, R=rough G=metal B=normal/occlusion mask.
            idx=(key//5)%4;blind=(key//19)%4;g=self.lib.gaxil[idx][blind].sample(imageuv,pix);rmw=self.lib.gaxil_rmw[idx].sample(imageuv,pix);N=self.lib.gaxil_normal[idx].sample(imageuv,pix);alpha=g[:,3:4]
            N=N*rmw[:,2:3]+np.array([.5,.5,1.])*(1-rmw[:,2:3]);front=material_response(g[:,:3],N,rmw[:,0],ratio,rmw[:,1]);c=c*(1-alpha)+front*alpha;e=e*(1-alpha+.12*alpha);lane='Gaxil/RMW+Three-room'
        elif lane=='three':
            idx=int(OVERLAYS[(key//5)%len(OVERLAYS)]);overlay=self.lib.overlay[idx].sample(imageuv,pix);alpha=overlay[:,3:4];c=c*(1-alpha)+overlay[:,:3]*alpha;e=e*(1-alpha+.16*alpha*overlay[:,:3]);lane='Three-Fenestra/layered'
        rgb[pane]=c;em[pane]=e
        if door or entry:
            handle=(np.abs(p[:,0]-.79)<.019)&(np.abs(p[:,1]-.46)<.035);rgb[handle]=srgb(np.array([.65,.65,.60]));em[handle]=0
        return rgb,em,lane
    def shade(self,s,r,maps,mode,dense=False):
        col,em=self.base(s,r,maps,mode,dense)
        if not len(s) or mode=='bare':return col,em
        k=r.get('accessory','');u,z=s.T;lib=self.lib
        if k.startswith('cutout_'):return col,em
        if r['id'] in self.roof_frames and not self.info[r['building']]['flat']:
            D=self.info[r['building']];xyz=np.c_[s,np.ones(len(s))]@self.roof_frames[r['id']];lo,hi=np.asarray(D['bounds']);x,y=xyz[:,:2].T;mid=(lo[1]+hi[1])/2
            xy=np.c_[(x-lo[0])/1.15,np.abs(y-mid)*1.09/1.6];mask=np.abs(y-mid)>.205
            for a,b,c,d in D['flash']:
                q=np.maximum(np.maximum(a-x,x-c),np.maximum(b-y,y-d));mask&=~((q>=0)&(q<.13))
            if mask.any():
                rgb=lib.slate.sample(xy[mask],self.ppm*1.15,True);n=lib.slate_normal.sample(xy[mask],self.ppm*1.15,True);col[mask]=material_response(rgb,n,np.full(len(rgb),.72),[0,0])
            return col,em
        if k=='stage21_glazed':
            w,h=r['span'];x,y=s.T/np.array([w,h])[:,None];key=self.seed(r['placement']);ratio,pix,ndv=self.angle(r,w,h)
            # Entire module has one frame/panel palette, never independent face states.
            low=(.27,.34,.40)[key%3];n=max(1,round(w/.60));edge=np.minimum(x,1-x)*w
            frame=(edge<.027)|(np.abs(x*n-np.rint(x*n))*w/n<.022)|(np.abs(y-low)*h<.032)|((1-y)*h<.038)
            glass=(y>=low)&~frame;panel=y<low
            palette=np.array([[.37,.28,.22],[.52,.53,.49],[.67,.66,.61],[.30,.35,.37],[.37,.42,.31]])
            col[panel]=srgb(palette[key%len(palette)]);col[panel&(np.mod(u/.12,1)<.08)]*=.79
            col[frame]=srgb(np.array([.77,.76,.72]) if key%4 else np.array([.40,.28,.20]))
            if glass.any():
                q=np.c_[x[glass],np.clip((y[glass]-low)/(1-low),0,1)];room=self.room(q,ratio,w,h*(1-low),pix,key);c,e=self.glass(room*.7,ndv,mode,key%100<25);col[glass]=c;em[glass]=e*.30
            return col,em
        if k=='vent' and 'span' in r:
            w,h=r['span'];hit=(u>w*.17)&(u<w*.83)&(z>h*.55)&(z<h*.83)
            if hit.any():
                q=np.c_[(u[hit]/w-.17)/.66,1-(z[hit]/h-.55)/.28];rgb=lib.vent.sample(q,self.ppm*w);N=lib.vent_normal.sample(q,self.ppm*w);col[hit]=material_response(rgb,N,np.full(len(q),.75),[0,0])
            return col,em
        if r['id'] not in self.rects:return col,em
        ops=self.rects[r['id']];pairs=self.pairs.get(str(r['id']),{});f=np.floor(z/r['floor_h']).astype(int);b=np.floor((u-r['first_center'])/r['spacing']+.5).astype(int);valid=(f>=0)&(f<r['floors'])&(b>=0)&(b<r['bays']);slot=np.where(valid,f*r['bays']+b,-1)
        style=r['building'].split(':')[-1]
        if style=='brick_stack':
            i=1;xy=np.c_[u/1.2,-z/1.2];N=lib.brick_normal[i].sample(xy,self.ppm*1.2,True);rough=lib.brick_rough[i].sample(xy,self.ppm*1.2,True)[:,0]
            col=material_response(col,N,rough,self.angle(r,1.2,1.2)[0])
        # Preserve stage23 weathering/joints for unchanged walls by using
        # authored color maps through the baseline maps argument. Repaint old
        # balcony door area that is no longer an aperture with same family.
        for j in np.unique(slot[valid]):
            j=int(j)
            if j not in ops:continue
            code,rect=ops[j];mask_slot=slot==j;key=self.seed(r['facade_id']+':'+str(j));pair=pairs.get(str(j))
            if pair:
                a,y,A,Y=rect;former=mask_slot&(u>=a)&(u<A)&(z>=y)&(z<Y)
                if former.any():
                    if style=='brick_stack':col[former]=lib.brick[1].sample(np.c_[u[former]/1.2,-z[former]/1.2],self.ppm*1.2,True)
                    elif style=='panel_block':col[former]=lib.concrete.sample(np.c_[u[former]/2.4,-z[former]/2.4],self.ppm*2.4,True)
                    else:col[former]=srgb(np.array([.66,.63,.54]))
                    em[former]=0
                x0=min(pair['door'][0],pair['window'][0]);x1=max(pair['door'][2],pair['window'][2]);shared=(x0,pair['door'][1],x1,pair['door'][3]);rectangles=[(2,pair['door']),(1,pair['window'])]
                a,y,A,Y=pair['post'];post=mask_slot&(u>=a)&(u<A)&(z>=y)&(z<Y);col[post]=srgb(np.array([.74,.73,.68]));em[post]=0
            else:rectangles=[(code&3,rect)];shared=None
            for role,R in rectangles:
                a,y,A,Y=R;hit=mask_slot&(u>=a)&(u<A)&(z>=y)&(z<Y)
                if hit.any():col[hit],em[hit],lane=self.opening(s[hit],r,R,key,mode,role,shared);self.counts[lane]=self.counts.get(lane,0)+int(hit.sum())
                if role==1:
                    sill=mask_slot&(u>=a-.025)&(u<A+.025)&(z>=y-.042)&(z<y)
                    col[sill]=srgb(np.array([.62,.64,.62]));em[sill]=0
        if r['door'] is not None:
            a,y,w,h=r['door'];R=(a,y,a+w,y+h);hit=(u>=a)&(u<a+w)&(z>=y)&(z<y+h)
            if hit.any():col[hit],em[hit],lane=self.opening(s[hit],r,R,self.seed(r['facade_id']+':entry'),mode,3);self.counts[lane]=self.counts.get(lane,0)+int(hit.sum())
        return col,em
