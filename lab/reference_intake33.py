"""Offline donor inventory and roughness candidate segmentation (no publishing).
glTF/GLB triangle modes, node transforms, UV0 and material bindings are read
directly. OBJ is supported through trimesh. Other formats are reported, not
silently treated as geometry. Source units are never asserted to be metres.
"""
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
import argparse, base64, hashlib, io, json, math, os, struct, tempfile, zipfile
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

TYPES={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}
WIDTH={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
LIMIT=2_000_000_000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')

def local_uri(root,uri):
    from urllib.parse import unquote,urlsplit
    u=urlsplit(uri)
    if u.scheme or u.netloc or u.query or u.fragment:raise ValueError('external URI prohibited')
    p=(root/unquote(u.path)).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():raise ValueError('missing or escaping URI')
    return p

def extract(zpath,root):
    with zipfile.ZipFile(zpath) as z:
        total=0;seen=set()
        for i in z.infolist():
            p=PurePosixPath(i.filename.replace(chr(92),'/'))
            if p.is_absolute() or '..' in p.parts or ':' in str(p) or (i.external_attr>>16)&0o170000==0o120000:
                raise ValueError('unsafe archive member')
            dest=(root/Path(*p.parts)).resolve()
            if not dest.is_relative_to(root.resolve()):raise ValueError('unsafe archive member')
            if i.is_dir():continue
            if dest in seen:raise ValueError('duplicate archive entry')
            seen.add(dest);total+=i.file_size
            if total>LIMIT:raise ValueError('archive limit')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as src,dest.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src,dst,1024*1024)

def triangulate(idx,mode):
    idx=np.asarray(idx)
    if idx.ndim!=1 or idx.dtype.kind not in 'ui':raise ValueError('index type')
    if mode==4:
        if len(idx)%3:raise ValueError('triangle count')
        return idx.reshape(-1,3)
    if mode==5:
        if len(idx)<3:return np.empty((0,3),dtype=idx.dtype)
        f=np.stack((idx[:-2],idx[1:-1],idx[2:]),axis=1)
        f[1::2]=f[1::2][:,[1,0,2]]
        return f
    if mode==6:
        if len(idx)<3:return np.empty((0,3),dtype=idx.dtype)
        return np.stack((np.full(len(idx)-2,idx[0],dtype=idx.dtype),idx[1:-1],idx[2:]),axis=1)
    raise ValueError('unsupported nontriangle mode')

def transform(n):
    if 'matrix' in n:return np.asarray(n['matrix'],float).reshape(4,4).T
    x,y,z,w=map(float,n.get('rotation',(0,0,0,1)))
    q=math.sqrt(x*x+y*y+z*z+w*w)
    if q<=0:raise ValueError('invalid rotation')
    x,y,z,w=[v/q for v in (x,y,z,w)]
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    M=np.eye(4);M[:3,:3]=R@np.diag(n.get('scale',(1,1,1)));M[:3,3]=n.get('translation',(0,0,0))
    return M

@dataclass
class Part:
    xyz: np.ndarray
    faces: np.ndarray
    uv: np.ndarray|None
    material: int|None
    node: str

class Gltf:
    def __init__(self,path):
        self.path=Path(path);self.root=self.path.parent;self.bin=None
        if self.path.suffix.lower()=='.glb':
            b=self.path.read_bytes();magic,version,size=struct.unpack_from('<4sII',b)
            if magic!=b'glTF' or version!=2 or size!=len(b):raise ValueError('GLB header')
            p=12;doc=None
            while p<len(b):
                n,t=struct.unpack_from('<II',b,p);p+=8
                if p+n>len(b):raise ValueError('GLB chunk bounds')
                if t==0x4e4f534a:doc=json.loads(b[p:p+n])
                elif t==0x004e4942:self.bin=b[p:p+n]
                p+=n
            if doc is None:raise ValueError('missing JSON')
            self.doc=doc
        else:self.doc=json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.buffers=[]
        for i,b in enumerate(self.doc.get('buffers',[])):
            if 'uri' not in b:
                if i or self.bin is None:raise ValueError('missing GLB buffer')
                data=self.bin
            else:
                uri=b['uri']
                if uri.startswith('data:'):
                    head,encoded=uri.split(',',1)
                    if not head.endswith(';base64'):raise ValueError('unsupported data URI')
                    data=base64.b64decode(encoded,validate=True)
                else:data=local_uri(self.root,uri).read_bytes()
            if len(data)<b['byteLength']:raise ValueError('truncated buffer')
            self.buffers.append(data[:b['byteLength']])
        self.materials=self.doc.get('materials',[])
        self.images=[]
        for i,img in enumerate(self.doc.get('images',[])):
            row={'index':i,'mime':img.get('mimeType'),'decoded':False}
            try:
                if 'bufferView' in img:data=self.view(img['bufferView'])
                elif img.get('uri','').startswith('data:'):data=base64.b64decode(img['uri'].split(',',1)[1],validate=True)
                else:data=local_uri(self.root,img['uri']).read_bytes()
                row['sha256']=hashlib.sha256(data).hexdigest()
                with Image.open(io.BytesIO(data)) as im:
                    row.update(size=list(im.size),format=im.format,mode=im.mode,decoded=True)
            except (OSError,ValueError,KeyError) as e:row['status']=type(e).__name__
            self.images.append(row)

    def view(self,i):
        v=self.doc['bufferViews'][i]
        if 'EXT_meshopt_compression' in v.get('extensions',{}):raise ValueError('meshopt compressed buffer unsupported')
        b=self.buffers[v['buffer']];start=v.get('byteOffset',0);end=start+v['byteLength']
        if start<0 or end>len(b):raise ValueError('view bounds')
        return b[start:end]

    def acc(self,i):
        a=self.doc['accessors'][i];dt=np.dtype(TYPES[a['componentType']]);w=WIDTH[a['type']];n=a['count']
        if n<0 or n*w*dt.itemsize>LIMIT:raise ValueError('accessor limit')
        def read(view,off,count,width,dtype,stride=None):
            data=self.view(view);stride=stride or width*dtype.itemsize
            if off<0 or stride<width*dtype.itemsize or (count and off+(count-1)*stride+width*dtype.itemsize>len(data)):raise ValueError('accessor bounds')
            return np.ndarray((count,width),dtype=dtype,buffer=data,offset=off,strides=(stride,dtype.itemsize)).copy()
        if 'bufferView' in a:
            v=a['bufferView'];x=read(v,a.get('byteOffset',0),n,w,dt,self.doc['bufferViews'][v].get('byteStride'))
        else:x=np.zeros((n,w),dt)
        if 'sparse' in a:
            s=a['sparse'];inds=s['indices'];vals=s['values'];k=s['count']
            ii=read(inds['bufferView'],inds.get('byteOffset',0),k,1,np.dtype(TYPES[inds['componentType']])).ravel()
            if len(ii) and (ii.max()>=n or np.any(ii[1:]<=ii[:-1])):raise ValueError('sparse index bounds/order')
            x[ii]=read(vals['bufferView'],vals.get('byteOffset',0),k,w,dt)
        if a.get('normalized') and dt.kind in 'ui':
            x=x.astype(np.float64)/np.iinfo(dt).max
            if dt.kind=='i':x=np.maximum(-1,x)
        return x

    def groups(self):
        d=self.doc;nodes=d.get('nodes',[]);meshes=d.get('meshes',[])
        if d.get('scenes'):roots=d['scenes'][d.get('scene',0)]['nodes']
        else:
            children={j for n in nodes for j in n.get('children',[])}
            roots=[i for i in range(len(nodes)) if i not in children]
        # Single pure scene-container levels are transparent, but preserve mesh groups.
        roots=[(i,np.eye(4),()) for i in roots]
        while len(roots)==1 and 'mesh' not in nodes[roots[0][0]] and nodes[roots[0][0]].get('children'):
            i,M,anc=roots[0]
            if i in anc:raise ValueError('cycle')
            roots=[(j,M@transform(nodes[i]),anc+(i,)) for j in nodes[i]['children']]
        def visit(i,M,anc,out):
            if i in anc:raise ValueError('node cycle')
            n=nodes[i];M=M@transform(n)
            if 'skin' in n:raise ValueError('skinned donor unsupported')
            if 'mesh' in n:
                for p in meshes[n['mesh']]['primitives']:
                    if p.get('extensions',{}).get('KHR_draco_mesh_compression'):raise ValueError('Draco unsupported')
                    if p.get('targets'):raise ValueError('morph targets unsupported')
                    pos=self.acc(p['attributes']['POSITION']).astype(np.float64)
                    if pos.shape[1]!=3 or not np.isfinite(pos).all():raise ValueError('position')
                    pos=pos@M[:3,:3].T+M[:3,3]
                    # glTF is right-handed Y-up; convert to PCG Z-up, determinant +1.
                    pos=pos[:,[0,2,1]];pos[:,1]*=-1
                    ii=self.acc(p['indices']).ravel() if 'indices' in p else np.arange(len(pos))
                    f=triangulate(ii,p.get('mode',4))
                    if len(f) and (f.max()>=len(pos) or f.min()<0):raise ValueError('index bounds')
                    if np.linalg.det(M[:3,:3])<0:f=f[:,[0,2,1]]
                    uv=self.acc(p['attributes']['TEXCOORD_0']) if 'TEXCOORD_0' in p['attributes'] else None
                    if uv is not None and (uv.shape!=(len(pos),2) or not np.isfinite(uv).all()):raise ValueError('UV0')
                    out.append(Part(pos,f,uv,p.get('material'),n.get('name',str(i))))
            for j in n.get('children',[]):visit(j,M,anc+(i,),out)
        for i,M,anc in roots:
            out=[];visit(i,M,anc,out)
            if out:yield nodes[i].get('name',str(i)),out

def obj_groups(path):
    import trimesh
    s=trimesh.load_scene(path,process=False)
    for n in sorted(s.graph.nodes_geometry):
        M,key=s.graph[n];g=s.geometry[key]
        xyz=np.asarray(g.vertices)@M[:3,:3].T+M[:3,3]
        f=np.asarray(g.faces)
        if np.linalg.det(M[:3,:3])<0:f=f[:,[0,2,1]]
        # OBJ axis and units are not standardized. User must confirm Z-up.
        uv=getattr(g.visual,'uv',None)
        yield str(n),[Part(xyz,f,None if uv is None else np.asarray(uv),None,str(n))]

def measure(parts):
    if not parts:raise ValueError('empty group')
    xyz=np.concatenate([p.xyz for p in parts]);lo=xyz.min(0);hi=xyz.max(0);span=hi-lo
    T=np.concatenate([p.xyz[p.faces] for p in parts])
    normal=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);area=np.linalg.norm(normal,axis=1)*.5
    scale=max(float(span.max()),1e-12);ok=area>scale*scale*1e-12
    T=T[ok];normal=normal[ok];area=area[ok]
    if not len(T):raise ValueError('no nondegenerate triangles')
    z=np.abs(normal[:,2])/(area*2);uv_tri=uv_nonzero=0;uv_out=0;ratios=[]
    for p in parts:
        if p.uv is None:continue
        t=p.uv[p.faces];uv_tri+=len(t)
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];ua=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5
        uv_nonzero+=int((ua>1e-12).sum());uv_out+=int(np.any((t<0)|(t>1),axis=(1,2)).sum())
        w=p.xyz[p.faces];ga=np.linalg.norm(np.cross(w[:,1]-w[:,0],w[:,2]-w[:,0]),axis=1)*.5
        mask=(ua>1e-12)&(ga>scale*scale*1e-12)
        ratios.extend((np.sqrt(ua[mask]/ga[mask])*scale).tolist())
    # UV scale is dimensionless relative to the group's largest extent.
    h=float(span[2]);levels=np.unique(np.round((xyz[:,2]-lo[2])/max(h,1e-12)*2048).astype(int))
    hist=np.zeros(2049);hist[np.clip(levels,0,2048)]=1
    scores=[]
    for floors in range(2,41):
        lag=round(2048/floors);a=hist[:-lag];b=hist[lag:]
        score=float((a*b).sum()/max(1.,np.sqrt(a.sum()*b.sum())))
        scores.append((score,floors))
    periods=[dict(floors_hypothesis=n,score=round(s,5),period_source_units=h/n) for s,n in sorted(scores,reverse=True)[:5]]
    hull=MultiPoint(xyz[:,:2]).convex_hull
    # Ground footprint is observed only when a near-ground horizontal surface
    # exists; otherwise convex hull is a proxy, not authoritative GIS.
    bottom=(z>.999)&(T[:,:,2].max(1)<=lo[2]+max(h*.002,scale*1e-6))
    polys=[Polygon(t[:,:2]) for t in T[bottom]]
    ground=unary_union([p for p in polys if p.area>scale*scale*1e-12]) if polys else None
    use_ground=ground is not None and ground.geom_type=='Polygon' and ground.is_valid and ground.area>hull.area*.15
    footprint=ground if use_ground else hull
    geometry={'bounds':[lo.tolist(),hi.tolist()],'span_source_units':span.tolist(),
      'triangles_input':int(len(ok)),'triangles_nondegenerate':int(ok.sum()),'degenerate':int((~ok).sum()),
      'surface_area_source_units2':float(area.sum()),'vertical_area_fraction':float(area[z<.05].sum()/area.sum()),
      'horizontal_area_fraction':float(area[z>.999].sum()/area.sum()),'sloped_area_fraction':float(area[(z>=.05)&(z<=.999)].sum()/area.sum()),
      'uv0_triangles':uv_tri,'uv0_nonzero_triangles':uv_nonzero,'uv0_outside_unit_square':uv_out,
      'uv_scale_relative_p10_p50_p90':np.quantile(ratios,[.1,.5,.9]).tolist() if ratios else None,
      'material_slots':sorted({p.material for p in parts if p.material is not None}),'period_hypotheses':periods,
      'footprint_kind':'near_ground_surface' if use_ground else 'convex_envelope_proxy',
      'footprint_area_source_units2':float(footprint.area)}
    return geometry,footprint,T

def material_inventory(g):
    doc=g.doc;tex=doc.get('textures',[])
    rows=[]
    for i,m in enumerate(g.materials):
        p=m.get('pbrMetallicRoughness',{})
        bindings={}
        for name,x in [('baseColor',p.get('baseColorTexture')),('metallicRoughness',p.get('metallicRoughnessTexture')),('normal',m.get('normalTexture')),('occlusion',m.get('occlusionTexture')),('emissive',m.get('emissiveTexture'))]:
            if x is None:continue
            t=tex[x['index']];source=t.get('source',t.get('extensions',{}).get('KHR_texture_basisu',{}).get('source'))
            bindings[name]={'texture':x['index'],'image':source,'texcoord':x.get('texCoord',0),'transform':x.get('extensions',{}).get('KHR_texture_transform')}
        rows.append(dict(index=i,name=m.get('name'),alpha=m.get('alphaMode','OPAQUE'),double_sided=m.get('doubleSided',False),bindings=bindings,base_color_factor=p.get('baseColorFactor',[1,1,1,1])))
    return rows

def audit_file(path,root,key,models,files):
    if path.suffix.lower() in ('.gltf','.glb'):
        g=Gltf(path);groups=g.groups();entry=dict(file=path.name,sha256=sha(path),axis='gltf Y-up -> Z-up',materials=material_inventory(g),images=g.images)
    else:
        groups=obj_groups(path);entry=dict(file=path.name,sha256=sha(path),axis='OBJ assumed Z-up; not certified',materials=[],images=[])
    for j,(name,parts) in enumerate(groups):
        k=f'{key}_{j:03d}';row,footprint,T=measure(parts)
        row.update(id=k,name=name,source=key,axis=entry['axis'])
        if footprint.geom_type=='Polygon':
            row['footprint']={'outer':list(footprint.exterior.coords)[:-1],'holes':[list(r.coords)[:-1] for r in footprint.interiors]}
        models.append(row)
    files.append(entry)

def run(low,high,out):
    out=Path(out)
    if os.getenv('GITHUB_ACTIONS')=='true' and os.getenv('PCG_PRIVATE_LOCAL')!='1':
        raise RuntimeError('private source audit prohibited in public CI')
    out.mkdir(parents=True,exist_ok=False)
    models=[];files=[];unsupported=[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);extract(Path(low),root)
        candidates=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.gltf','.glb','.obj'))
        unsupported=sorted({p.suffix.lower() for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.fbx','.dae','.blend','.mesh','.max','.3ds')})
        if not candidates:raise ValueError('no supported geometry in low archive; inspect format')
        for i,p in enumerate(candidates):audit_file(p,root,f'low_{i:03d}',models,files)
        audit_file(Path(high),Path(high).parent,'high',models,files)
    maps=[];roughness_candidates=[]
    for p in sorted(Path(high).parent.iterdir()):
        if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.ktx2') and ('_RBP_' in p.name):
            r={'name':p.name,'sha256':sha(p),'bound':False}
            try:
                with Image.open(p) as im:r.update(size=list(im.size),mode=im.mode,format=im.format)
            except OSError:r['status']='decode unsupported'
            maps.append(r)
            if 'roughness' in p.name.lower() and r.get('format') in ('JPEG','PNG'):
                from reference_atlas33 import analyze
                roughness_candidates.append(dict(file=p.name,analysis=analyze(p,out)))
    report=dict(schema='pcg.donor.audit.v1',low_sha256=sha(low),high_sha256=sha(high),objects=models,files=files,external_unassigned_maps=maps,roughness_candidates=roughness_candidates,unsupported_archive_formats=unsupported,
       limitations=['No claim of metric scale without independent calibration.','Floor periods are hypotheses, not classifications.','UV0 coverage counts, not UV island semantics or baking.','Unbound external maps are not assigned automatically.','No licensed source data uploaded.'])
    dump(out/'catalog.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--low',type=Path,required=True);p.add_argument('--high',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=run(a.low,a.high,a.out)
    print('PRIVATE_CATALOG_READY',len(r['objects']),flush=True)
if __name__=='__main__':main()
