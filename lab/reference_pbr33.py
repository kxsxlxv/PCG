"""Stage 33: measured facade transfer and floor-strip reuse.
Diagnostic per-building directional atlas, not production shared atlas.
External normal convention +Y and material assignment are hypotheses.
"""
import argparse,io,json,struct
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import cv2
from reference_intake33 import Gltf,dump
from reference_match33 import parts_arrays,frame,mesh_tiers
from reference_contours33 import bounded_mesh

VIEWS=np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])
def basis(view):
    v=np.asarray(view,float);v/=np.linalg.norm(v)
    r=np.cross([0,0,1.],v)
    if np.linalg.norm(r)<1e-8:r=np.array([1.,0,0])
    else:r/=np.linalg.norm(r)
    return r,np.cross(v,r),v

def project(p,center,extent,view):
    r,u,v=basis(view);q=p-center
    return np.stack((.5+(q@r)/(2*extent[0]),.5-(q@u)/(2*extent[1])),axis=-1)

def normals(t):
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    return n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)

def tangents(t,uv,n):
    d=t[:,1:]-t[:,:1];q=uv[:,1:]-uv[:,:1]
    det=q[:,0,0]*q[:,1,1]-q[:,0,1]*q[:,1,0]
    assert np.all(abs(det)>1e-14)
    T=(d[:,0]*q[:,1,1,None]-d[:,1]*q[:,0,1,None])/det[:,None]
    B=(d[:,1]*q[:,0,0,None]-d[:,0]*q[:,1,0,None])/det[:,None]
    T-=n*np.sum(T*n,axis=1)[:,None]
    T/=np.maximum(np.linalg.norm(T,axis=1)[:,None],1e-20)
    sign=np.where(np.sum(np.cross(n,T)*B,axis=1)<0,-1.,1.)
    return np.c_[T,sign]

def fill_nearest(a,mask,default):
    if not mask.any():return np.broadcast_to(np.array(default,np.uint8),a.shape).copy()
    if mask.all():return a.copy()
    _,labels=cv2.distanceTransformWithLabels((~mask).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    table=np.zeros((int(labels.max())+1,a.shape[-1]),np.uint8)
    table[labels[mask]]=a[mask]
    return table[labels]

def floor_strips(rough,mask):
    """Estimate row pitch from projected, visible low-roughness pixels.
    Require three repetitions; strips are hypotheses, not semantic floor IDs."""
    ys=np.flatnonzero(mask.any(axis=1))
    if len(ys)<100:return None
    y0,y1=int(ys[0]),int(ys[-1]+1);h=y1-y0
    glass=(rough[:,:,1]<32)&mask
    # Remove tiny flecks and use horizontal population normalized by visible width.
    glass=cv2.morphologyEx(glass.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
    sig=glass.sum(1)/np.maximum(mask.sum(1),1)
    sig=sig[y0:y1]
    if sig.mean()<.015 or sig.std()<.012:return None
    sig=cv2.GaussianBlur(sig[:,None].astype('f4'),(1,5),0).ravel()
    x=sig-sig.mean()
    lo=max(8,int(h*.035));hi=min(h//3,int(h*.30))
    scores=[]
    for k in range(lo,hi+1):
        a,b=x[:-k],x[k:];s=float(a@b/max(1e-9,np.linalg.norm(a)*np.linalg.norm(b)))
        scores.append((k,s))
    peaks=[(k,s) for i,(k,s) in enumerate(scores) if 0<i<len(scores)-1 and s>=scores[i-1][1] and s>=scores[i+1][1] and s>=.42]
    if not peaks:return None
    best=max(s for k,s in peaks);k,s=min((k,s) for k,s in peaks if s>=max(.42,best*.86))
    # Prefer cuts between rows of glass; preserve base and crown bands.
    cuts=[]
    for phase in range(k):
        a=np.arange(y0+phase,y1,k)
        a=a[(a>=y0+k*.45)&(a<=y1-k*.45)]
        if len(a)>=4:cuts.append((float(np.mean(glass[a].sum(1)/np.maximum(mask[a].sum(1),1))),a))
    if not cuts:return None
    score,a=min(cuts,key=lambda q:q[0])
    if score>max(.025,float(sig.mean())*.55):return None
    return dict(pitch_px=int(k),correlation=float(s),cuts=[int(z) for z in a],glass_cut_fraction=score,visible_bounds=[y0,y1],estimated_period_fraction=float(k/h))

def recompose(maps,strip,seed,mask):
    # Reorder only fully observed bands with similar silhouettes. Same permutation
    # in all three material maps; roof and ground unchanged.
    if strip is None:return maps,0
    cuts=strip['cuts'];pairs=list(zip(cuts[:-1],cuts[1:]))
    if len(pairs)<3:return maps,0
    rng=np.random.default_rng(seed);perm=rng.permutation(len(pairs))
    out=[a.copy() for a in maps];count=0
    for i,j in enumerate(perm):
        if i==j:continue
        y,Y=pairs[i];a,b=pairs[int(j)]
        if Y-y!=b-a:continue
        ma,mb=mask[y:Y],mask[a:b]
        if (ma&mb).sum()/max(1,(ma|mb).sum())<.90:continue
        for dst,src in zip(out,maps):dst[y:Y]=src[a:b]
        count+=1
    return out,count

VS="""#version 330
in vec3 p;in vec3 n;in vec2 uv;
uniform vec3 center,right,up,view;uniform vec2 extent;uniform float depth;
out vec3 P;out vec3 N;out vec2 U;
void main(){vec3 q=p-center;gl_Position=vec4(dot(q,right)/extent.x,dot(q,up)/extent.y,-dot(q,view)/depth,1);P=p;N=n;U=uv;}
"""
FS="""#version 330
in vec3 P;in vec3 N;in vec2 U;
uniform sampler2D albedo,normalMap,ormMap;
uniform vec3 right,up,view;uniform int mode;
layout(location=0) out vec4 A;layout(location=1) out vec4 B;layout(location=2) out vec4 C;
vec3 normal_world(){
vec3 n=normalize(N);if(dot(n,view)<0.)n=-n;
vec3 dx=dFdx(P),dy=dFdy(P);vec2 ux=dFdx(U),uy=dFdy(U);
float det=ux.x*uy.y-ux.y*uy.x;
if(abs(det)<1e-14)return n;
vec3 t=(dx*uy.y-dy*ux.y)/det;vec3 b=(dy*ux.x-dx*uy.x)/det;
t=normalize(t-n*dot(t,n));b=normalize(cross(n,t))*sign(dot(cross(n,t),b));
vec3 m=texture(normalMap,U).rgb*2.-1.;
return normalize(t*m.x+b*m.y+n*m.z);
}
void main(){
vec3 a=texture(albedo,U).rgb;vec3 n=normal_world();float rough=texture(ormMap,U).g;
if(mode==0){
 A=vec4(a,1);B=vec4(vec3(dot(n,right),dot(n,-up),max(0.,dot(n,view)))*.5+.5,1);
 C=vec4(1.,rough,0.,1.);return;
}
vec3 L=normalize(vec3(.4,-.5,.75));vec3 H=normalize(L+view);
float nl=max(0.,dot(n,L)),nv=max(.02,dot(n,view)),nh=max(0.,dot(n,H)),vh=max(0.,dot(view,H));
float r=max(.06,rough),aa=r*r;aa*=aa;
float D=aa/(3.14159265*pow(nh*nh*(aa-1.)+1.,2.)+.00001);
float k=(r+1.)*(r+1.)/8.;
float G=nl/(nl*(1.-k)+k)*nv/(nv*(1.-k)+k);
float F=.04+.96*pow(1.-vh,5.);
float spec=D*G*F/(4.*max(nl,.02)*nv);
vec3 lin=pow(max(a,vec3(0)),vec3(2.2));
vec3 c=lin*(.32+.68*nl)+vec3(min(spec*nl,.6))*.30;
A=vec4(pow(clamp(c,0.,1.),vec3(1./2.2)),1);B=vec4(0);C=vec4(0);
}
"""
class BakeRenderer:
    def __init__(self):
        import moderngl,ctypes.util
        self.ctx=moderngl.create_context(standalone=True,backend='egl',require=330,libegl=ctypes.util.find_library('EGL'),libgl=ctypes.util.find_library('GL'))
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.p=self.ctx.program(vertex_shader=VS,fragment_shader=FS)
        for i,k in enumerate(('albedo','normalMap','ormMap')):self.p[k].value=i
        self.tex=[]
    def maps(self,arrays):
        for t in self.tex:t.release()
        self.tex=[]
        for i,a in enumerate(arrays):
            t=self.ctx.texture((a.shape[1],a.shape[0]),3,np.ascontiguousarray(a).tobytes(),alignment=1)
            t.repeat_x=False;t.repeat_y=False;t.use(i);self.tex.append(t)
    def draw(self,t,uv,center,extent,view,mode,res):
        r,u,v=basis(view);n=normals(t)
        data=np.c_[t.reshape(-1,3),np.repeat(n,3,axis=0),np.nan_to_num(uv).reshape(-1,2)].astype('f4')
        b=self.ctx.buffer(data.tobytes());vao=self.ctx.vertex_array(self.p,[(b,'3f 3f 2f','p','n','uv')])
        tex=[self.ctx.texture(res,4) for _ in range(3)];depth=self.ctx.depth_renderbuffer(res)
        f=self.ctx.framebuffer(tex,depth);f.use();f.clear(.91,.93,.95,0.,depth=1.)
        for k,val in dict(center=center,right=r,up=u,view=v,extent=extent,depth=float(max(extent)*4),mode=mode).items():
            self.p[k].value=tuple(val) if np.ndim(val) else val
        vao.render()
        arr=[np.frombuffer(f.read(attachment=i,components=4,alignment=1),np.uint8).reshape(res[1],res[0],4)[::-1].copy() for i in range(3)]
        f.release();depth.release();vao.release();b.release()
        for q in tex:q.release()
        return arr

def side_extents(span,view):
    r,u,v=basis(view);return np.array([abs(r)@span,abs(u)@span])*.515

def bake(renderer,t,uv,g,span,maps,size,seed):
    renderer.maps(maps);center=span/2;n=normals(g)
    side=np.argmax(n@VIEWS.T,axis=1)
    gu=np.zeros(g.shape[:2]+(2,))
    atl=[np.zeros((size*2,size*3,3),np.uint8) for _ in range(3)]
    alt=[a.copy() for a in atl];report=[]
    total_hits=total_count=swapped=0
    for i,view in enumerate(VIEWS):
        ext=side_extents(span,view);images=renderer.draw(t,uv,center,ext,view,0,(size,size))
        mask=images[0][:,:,3]>0
        tiles=[fill_nearest(a[:,:,:3],mask,d) for a,d in zip(images,([150,150,150],[128,128,255],[255,180,0]))]
        strip=floor_strips(tiles[2],mask) if i<4 else None
        variants,count=recompose(tiles,strip,seed+i,mask);swapped+=count
        y,x=divmod(i,3)
        for dst,variant,a,b in zip(atl,alt,tiles,variants):
            dst[y*size:(y+1)*size,x*size:(x+1)*size]=a
            variant[y*size:(y+1)*size,x*size:(x+1)*size]=b
        hit=side==i;q=project(g[hit],center,ext,view)
        # Sample target triangle interior, not just vertices, for projection coverage.
        samples=np.array([[1/3,1/3,1/3],[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.45,.45,.10],[.45,.10,.45],[.10,.45,.45]])
        pts=np.einsum('sj,tjk->tsk',samples,q)
        xy=np.clip((pts*size).astype(int),0,size-1)
        hits=int(mask[xy[:,:,1],xy[:,:,0]].sum());num=xy.shape[0]*xy.shape[1]
        total_hits+=hits;total_count+=num
        gu[hit]=(np.clip(q,0,1)+[x,y])/[3,2]
        glass=(tiles[2][:,:,1]<32)&mask
        num_labels,labels,stats,cent=cv2.connectedComponentsWithStats(glass.astype(np.uint8),8)
        regions=[dict(rect_px=[int(v) for v in q[:4]],area_px=int(q[4])) for q in stats[1:] if q[2]>=3 and q[3]>=3 and q[4]>=12]
        report.append(dict(side=i,projected_pixels=int(mask.sum()),triangle_samples=num,valid_samples=hits,glass_candidates=regions,floor_strip=strip,reordered_bands=count))
    return atl,alt,gu,dict(sides=report,projection_sample_coverage=total_hits/max(1,total_count),reordered_bands=swapped)

def glb_write(path,t,uv,maps):
    """Explicit glTF Y-up, UV top-left, embedded textures, tangent handedness."""
    n=normals(t);tan=tangents(t,uv,n)
    def y_up(a):return a[..., [0,2,1]]*np.array([1,1,-1])
    pos=y_up(t.reshape(-1,3)).astype('<f4');nor=y_up(np.repeat(n,3,axis=0)).astype('<f4')
    tt=np.repeat(np.c_[y_up(tan[:,:3]),tan[:,3]],3,axis=0).astype('<f4')
    u=uv.reshape(-1,2).astype('<f4');idx=np.arange(len(pos),dtype='<u4')
    blob=bytearray();views=[];access=[]
    def put(data,target=None):
        blob.extend(b'\0'*(-len(blob)%4));o=len(blob);blob.extend(data)
        d=dict(buffer=0,byteOffset=o,byteLength=len(data))
        if target:d['target']=target
        views.append(d);return len(views)-1
    def acc(a,typ,component,target,bounds=False):
        d=dict(bufferView=put(a.tobytes(),target),componentType=component,count=len(a),type=typ)
        if bounds:d.update(min=a.min(0).tolist(),max=a.max(0).tolist())
        access.append(d);return len(access)-1
    attrs=dict(POSITION=acc(pos,'VEC3',5126,34962,True),NORMAL=acc(nor,'VEC3',5126,34962),TANGENT=acc(tt,'VEC4',5126,34962),TEXCOORD_0=acc(u,'VEC2',5126,34962))
    ix=acc(idx,'SCALAR',5125,34963)
    images=[]
    for a in maps:
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');images.append(dict(bufferView=put(b.getvalue()),mimeType='image/png'))
    doc=dict(asset={'version':'2.0','generator':'PCG measured facade diagnostic'},buffers=[{'byteLength':len(blob)}],bufferViews=views,accessors=access,images=images,
       samplers=[dict(magFilter=9729,minFilter=9729,wrapS=33071,wrapT=33071)],textures=[dict(sampler=0,source=i) for i in range(3)],
       materials=[dict(name='projected_donor_pbr',pbrMetallicRoughness=dict(baseColorTexture={'index':0},metallicRoughnessTexture={'index':2},metallicFactor=0,roughnessFactor=1),normalTexture={'index':1})],
       meshes=[dict(primitives=[dict(attributes=attrs,indices=ix,material=0)])],nodes=[{'mesh':0}],scenes=[{'nodes':[0]}],scene=0)
    j=json.dumps(doc,separators=(',',':')).encode();j+=b' '*(-len(j)%4);blob.extend(b'\0'*(-len(blob)%4))
    path.write_bytes(struct.pack('<4sII','glTF'.encode(),2,12+8+len(j)+8+len(blob))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(blob),0x004e4942)+blob)
    check=Gltf(path);parts=list(check.groups())[0][1]
    assert len(parts)==1 and len(parts[0].faces)==len(t)
    np.testing.assert_allclose(parts[0].xyz[parts[0].faces],t,atol=2e-5,rtol=1e-6)
    np.testing.assert_allclose(parts[0].uv[parts[0].faces],uv,atol=1e-7)
    assert len(check.images)==3 and all(i['decoded'] for i in check.images)
    return parts[0].xyz[parts[0].faces],parts[0].uv[parts[0].faces]

def run(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    ref=json.loads((args.reference/'report.json').read_text())
    rows={r['id']:r for r in ref['models']};selected={r['id']:r for r in ref['comparisons'] if r['pack']=='high'}
    a=np.asarray(Image.open(args.albedo).convert('RGB'));n=np.asarray(Image.open(args.normal).convert('RGB'));r=np.asarray(Image.open(args.rough).convert('L'))
    assert a.shape[:2]==r.shape
    orm=np.zeros(a.shape,np.uint8);orm[:,:,0]=255;orm[:,:,1]=r
    source=[a,n,orm];renderer=BakeRenderer();results=[]
    for i,(_,parts) in enumerate(Gltf(args.high).groups()):
        k=f'high_{i:03d}'
        if k not in selected:continue
        t,uv,norm,area=parts_arrays(parts);t,norm,span,theta,confidence=frame(t,norm,area)
        row=rows[k];assert len(t)==selected[k]['source_triangles']
        fit=bounded_mesh(row['contours'],span) if row['contours'] else None
        g,_,_,_=fit if fit is not None else (*mesh_tiers(row['tiers'],span),None)
        assert len(g)==selected[k]['grammar_triangles']
        atlas,variant,gu,meta=bake(renderer,t,uv,g,span,source,args.size,33000+i)
        g,gu=glb_write(out/(k+'_pbr.glb'),g,gu,atlas)
        if meta['reordered_bands']:glb_write(out/(k+'_recomposed.glb'),g,gu,variant)
        res=(1280,960);extent=(float(np.linalg.norm(span)*.60*res[0]/res[1]),float(np.linalg.norm(span)*.60));center=span/2
        scores=[]
        sheet=Image.new('RGB',(res[0]*3,(res[1]+35)*2),(235,237,240));d=ImageDraw.Draw(sheet)
        for j,view in enumerate(((1,-1.7,.65),(-1,1.7,.65))):
            ims=[]
            for geom,tex,ma in ((t,uv,source),(g,gu,atlas),(g,gu,variant)):
                renderer.maps(ma);im=renderer.draw(geom,tex,center,extent,view,1,res)[0];ims.append(im)
            A,B=ims[:2];m=A[:,:,3]>0;q=B[:,:,3]>0;common=m&q
            scores.append(dict(iou=float(common.sum()/max(1,(m|q).sum())),rgb_mae=float(np.abs(A[:,:,:3].astype(float)-B[:,:,:3])[common].mean()/255) if common.any() else 1.))
            for l,im in enumerate(ims):
                sheet.paste(Image.fromarray(im).convert('RGB'),(l*res[0],j*(res[1]+35)))
                d.text((l*res[0]+12,j*(res[1]+35)+res[1]+7),('Source / external PBR' if l==0 else 'Generated shell / projected PBR' if l==1 else 'Reordered floor strips' if meta['reordered_bands'] else 'No reliable strip variation')+f' | view {j+1}',fill='black')
        sheet.save(out/(k+'_pbr_comparison.png'))
        dump(out/(k+'_facades.json'),meta)
        results.append(dict(id=k,triangles=len(g),source_triangles=len(t),scores=scores,coverage=meta['projection_sample_coverage'],reordered_bands=meta['reordered_bands'],sides_with_period=sum(x['floor_strip'] is not None for x in meta['sides']),glass_candidates=sum(len(x['glass_candidates']) for x in meta['sides'])))
    assert len(results)==len(selected)
    report=dict(models=results,source_maps='external atlas, assumed +Y normal, no verified author binding',limitations=['Directional projection: recesses and repeated depth layers can smear or duplicate details.','Per-building diagnostic texture atlas, not shared production module atlas.','Floor strip variations are projected image evidence, not classified windows or GIS-ready grammar.','Direct GGX lighting approximation in software EGL, not UNIGINE.','Low-poly shapes retained from preceding experiment; no low-poly PBR binding validation.'])
    dump(out/'report.json',report);print('PBR33_RESULT',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('high','albedo','normal','rough','reference','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--size',type=int,default=768);run(p.parse_args())
