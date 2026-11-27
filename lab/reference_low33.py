"""Native Low-poly material binding audit and matched diagnostic projection.
Uses actual glTF material IDs and embedded images, no external high atlas.
Unbound normal/roughness are explicitly neutral/factor-only, not invented maps.
"""
import argparse,base64,io,json,tempfile,ctypes.util
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from reference_intake33 import Gltf,extract,local_uri,dump
from reference_match33 import parts_arrays,frame,mesh_tiers
from reference_contours33 import bounded_mesh
from reference_pbr33 import BakeRenderer,VS,FS,basis,normals,bake,glb_write

def image(g,i):
    q=g.doc['images'][i]
    if 'bufferView' in q:data=g.view(q['bufferView'])
    elif q.get('uri','').startswith('data:'):data=base64.b64decode(q['uri'].split(',',1)[1])
    else:data=local_uri(g.root,q['uri']).read_bytes()
    return np.asarray(Image.open(io.BytesIO(data)).convert('RGBA')).copy()

def material(g,i):
    q=g.doc['materials'][i] if i is not None else {}
    p=q.get('pbrMetallicRoughness',{})
    mode=q.get('alphaMode','OPAQUE')
    if mode=='BLEND':raise ValueError('BLEND source not supported by diagnostic raster')
    def tex(info):
        if not info:return None,(False,False)
        if info.get('texCoord',0)!=0 or info.get('extensions',{}).get('KHR_texture_transform'):raise ValueError('unsupported transformed/secondary UV')
        t=g.doc['textures'][info['index']]
        s=g.doc.get('samplers',[])[t['sampler']] if 'sampler' in t else {}
        wraps=[s.get(k,10497) for k in ('wrapS','wrapT')]
        if any(w not in (10497,33071) for w in wraps):raise ValueError('unsupported mirrored wrap')
        return image(g,t['source']),tuple(w==10497 for w in wraps)
    a,aw=tex(p.get('baseColorTexture'));n,nw=tex(q.get('normalTexture'));r,rw=tex(p.get('metallicRoughnessTexture'))
    bindings=dict(basecolor=a is not None,normal=n is not None,metallic_roughness=r is not None)
    if a is None:a=np.full((2,2,4),255,np.uint8)
    factor=np.array(p.get('baseColorFactor',[1,1,1,1]))
    a=a.copy();a[:,:,:3]=np.clip(a[:,:,:3]*factor[:3]**(1/2.2),0,255).astype(np.uint8);a[:,:,3]=np.clip(a[:,:,3]*factor[3],0,255).astype(np.uint8)
    if n is None:n=np.broadcast_to([128,128,255],(2,2,3)).astype(np.uint8).copy()
    else:
        n=n[:,:,:3].copy();v=n.astype(float)/127.5-1.;v[:,:,:2]*=float(q.get('normalTexture',{}).get('scale',1.))
        v/=np.maximum(np.linalg.norm(v,axis=2,keepdims=True),1e-8);n=np.clip((v+1)*127.5,0,255).astype(np.uint8)
    rough=float(p.get('roughnessFactor',1.))
    if r is None:r=np.full((2,2,3),255,np.uint8)
    else:r=r[:,:,:3].copy()
    r[:,:,0]=255;r[:,:,1]=np.clip(r[:,:,1]*rough,0,255).astype(np.uint8);r[:,:,2]=0
    return dict(maps=[a,n,r],wrap=[aw,nw,rw],alpha_cut=float(q.get('alphaCutoff',.5)) if mode=='MASK' else -1.,bindings=bindings,alpha_mode=mode,metallic_ignored=float(p.get('metallicFactor',1.)))

class NativeRenderer(BakeRenderer):
    def __init__(self):
        super().__init__();self.p.release()
        frag=FS.replace('uniform int mode;','uniform int mode; uniform float alphaCut;').replace('vec3 a=texture(albedo,U).rgb;','vec4 raw=texture(albedo,U);if(alphaCut>=0.&&raw.a<alphaCut)discard;vec3 a=raw.rgb;')
        self.p=self.ctx.program(vertex_shader=VS,fragment_shader=frag)
        for i,k in enumerate(('albedo','normalMap','ormMap')):self.p[k].value=i
        self.source=None;self.batches=[]
    def maps(self,arrays,wrap=None):
        for t in self.tex:t.release()
        self.tex=[]
        for i,a in enumerate(arrays):
            t=self.ctx.texture((a.shape[1],a.shape[0]),a.shape[2],np.ascontiguousarray(a).tobytes(),alignment=1)
            t.repeat_x,t.repeat_y=(False,False) if wrap is None else wrap[i]
            t.use(i);self.tex.append(t)
    def draw(self,t,uv,center,extent,view,mode,res):
        if t is not self.source:
            self.p['alphaCut'].value=-1.
            return super().draw(t,uv,center,extent,view,mode,res)
        r,u,v=basis(view);tex=[self.ctx.texture(res,4) for _ in range(3)];depth=self.ctx.depth_renderbuffer(res)
        f=self.ctx.framebuffer(tex,depth);f.use();f.clear(.91,.93,.95,0.,depth=1.)
        for k,val in dict(center=center,right=r,up=u,view=v,extent=extent,depth=float(max(extent)*4),mode=mode).items():self.p[k].value=tuple(val) if np.ndim(val) else val
        for indices,mat in self.batches:
            self.maps(mat['maps'],mat['wrap']);self.p['alphaCut'].value=mat['alpha_cut']
            q=t[indices];n=normals(q);u=uv[indices]
            data=np.c_[q.reshape(-1,3),np.repeat(n,3,axis=0),np.nan_to_num(u).reshape(-1,2)].astype('f4')
            b=self.ctx.buffer(data.tobytes());vao=self.ctx.vertex_array(self.p,[(b,'3f 3f 2f','p','n','uv')]);vao.render();vao.release();b.release()
        arr=[np.frombuffer(f.read(attachment=i,components=4,alignment=1),np.uint8).reshape(res[1],res[0],4)[::-1].copy() for i in range(3)]
        f.release();depth.release()
        for q in tex:q.release()
        return arr

def indices_by_material(parts):
    T=np.concatenate([p.xyz[p.faces] for p in parts]);ids=np.concatenate([np.full(len(p.faces),-1 if p.material is None else p.material,int) for p in parts])
    span=np.ptp(T.reshape(-1,3),axis=0);n=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);a=np.linalg.norm(n,axis=1)
    ok=a>2*max(float(span.max()),1e-12)**2*1e-12;ids=ids[ok]
    return {int(i):np.flatnonzero(ids==i) for i in np.unique(ids)}

def run(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    ref=json.loads((args.reference/'report.json').read_text());rows={r['id']:r for r in ref['models']};selected={r['id']:r for r in ref['comparisons'] if r['pack']=='low'}
    renderer=NativeRenderer();results=[];skipped=[];inventory=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);extract(args.low,root)
        paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.gltf','.glb','.obj'))
        for j,path in enumerate(paths):
            if path.suffix.lower()=='.obj':continue
            g=Gltf(path);cache={}
            for i,(_,parts) in enumerate(g.groups()):
                k=f'low_{j:03d}_{i:03d}'
                if k not in selected:continue
                t,uv,n,a=parts_arrays(parts);t,n,span,theta,conf=frame(t,n,a);assert len(t)==selected[k]['source_triangles']
                ix=indices_by_material(parts);assert sum(map(len,ix.values()))==len(t)
                try:
                    for m in ix:
                        if m not in cache:cache[m]=material(g,None if m==-1 else m)
                except (ValueError,KeyError) as e:
                    skipped.append(dict(id=k,reason=str(e)));continue
                if not np.isfinite(uv).all():skipped.append(dict(id=k,reason='missing UV0'));continue
                renderer.source=t;renderer.batches=[(ii,cache[m]) for m,ii in ix.items()]
                row=rows[k];fit=bounded_mesh(row['contours'],span) if row['contours'] else None
                shell,_,_,_=fit if fit is not None else (*mesh_tiers(row['tiers'],span),None)
                assert len(shell)==selected[k]['grammar_triangles']
                # draw() selects native per-primitive material batches for source t.
                placeholder=[np.full((2,2,3),128,np.uint8)]*3
                atlas,_,su,meta=bake(renderer,t,uv,shell,span,placeholder,args.size,33100+i)
                shell,su=glb_write(out/(k+'_native_baked.glb'),shell,su,atlas)
                res=(1280,960);ext=float(np.linalg.norm(span)*.60);center=span/2
                sheet=Image.new('RGB',(res[0]*2,(res[1]+35)*2),(235,237,240));d=ImageDraw.Draw(sheet);scores=[]
                for v,view in enumerate(((1,-1.7,.65),(-1,1.7,.65))):
                    A=renderer.draw(t,uv,center,(ext*4/3,ext),view,1,res)[0]
                    renderer.maps(atlas);B=renderer.draw(shell,su,center,(ext*4/3,ext),view,1,res)[0]
                    a=A[:,:,3]>0;b=B[:,:,3]>0;m=a&b
                    scores.append(dict(iou=float(m.sum()/max(1,(a|b).sum())),rgb_mae=float(np.abs(A[:,:,:3].astype(float)-B[:,:,:3])[m].mean()/255) if m.any() else 1.))
                    for l,im in enumerate((A,B)):
                        sheet.paste(Image.fromarray(im).convert('RGB'),(l*res[0],v*(res[1]+35)));d.text((l*res[0]+10,v*(res[1]+35)+res[1]+7),('Native embedded materials' if l==0 else 'Independent shell / baked native material')+f' | view {v+1}',fill='black')
                sheet.save(out/(k+'_native_comparison.png'))
                results.append(dict(id=k,source_triangles=len(t),triangles=len(shell),scores=scores,coverage=meta['projection_sample_coverage'],materials=[dict(id=m,**{x:cache[m][x] for x in ('bindings','alpha_mode','metallic_ignored')}) for m in ix]))
            inventory.append(dict(file=path.name,materials=len(g.materials),images=len(g.images),decoded=sum(x['decoded'] for x in g.images)))
    assert len(results)+len(skipped)==len(selected)
    assert len(results)>=4
    report=dict(models=results,skipped=skipped,inventory=inventory,limitations=['Native texture bindings and UV0 used; absent normal or roughness textures use neutral or factor-only material.','Metallic, occlusion and blend materials not reproduced by diagnostic lighting.','Directional projection can smear occluded surfaces; not shared module library.','Only six matched samples, not all low-poly objects.'])
    dump(out/'report.json',report);print('LOW33_RESULT',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('low','reference','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--size',type=int,default=768);run(p.parse_args())
