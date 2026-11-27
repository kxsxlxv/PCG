"""Private, offline PBR conversion for the stage-28 material bank.
No downloaded data is committed. Sources are explicit and fail closed.
"""
from pathlib import Path
import hashlib, io, json, os, zipfile
import numpy as np
from PIL import Image

SOURCES = {
    'brick_light':'Bricks060_2K-PNG.zip', 'brick_red':'Bricks077_2K-PNG.zip',
    'concrete':'Concrete034_2K-PNG.zip', 'granite':'Granite002A_2K-PNG.zip',
    'painted_metal':'PaintedMetal010_2K-PNG.zip', 'painted_plaster':'PaintedPlaster015_2K-PNG.zip',
    'wood':'PaintedWood001_2K-PNG.zip', 'paving':'PavingStones143_2K-PNG.zip',
    'plaster':'Plaster001_2K-PNG.zip', 'clay':'RoofingTiles012A_2K-PNG.zip',
    'aged_clay':'RoofingTiles013C_2K-PNG.zip', 'bitumen':'bitumen_2k.zip',
    'tile_facade':'concrete_tile_facade_2k.zip', 'corrugated':'corrugated_iron_2k.zip',
}
# The physical repeats below are authored estimates, not measured source dimensions.
# No change to historical IDs. Timber keeps procedural board joints; the source
# contributes paint microdetail only. Preserve two standing-seam variants.
# layer: (source, repeat meters, linear RGB multiplier, mode)
SLOTS = {
    16:('brick_red',2.4,(1.,1.,1.),'full'),
    17:('brick_red',2.4,(.68,.64,.62),'full'),
    18:('brick_light',1.2,(1.,.86,.68),'full'),
    19:('wood',1.52,(1.,1.,1.),'hybrid'),
    20:('wood',1.52,(1.,1.,1.),'hybrid'),
    21:('wood',1.52,(1.,1.,1.),'hybrid'),
    22:('plaster',2.,(1.,.87,.70),'full'),
    23:('plaster',2.,(1.05,1.05,1.05),'full'),
    24:('painted_plaster',2.,(1.,.95,.87),'full'),
    25:('concrete',2.4,(.85,.85,.85),'full'),
    26:('concrete',2.4,(1.,.93,.79),'full'),
    27:('concrete',2.4,(1.13,1.13,1.13),'full'),
    29:('corrugated',2.4,(.35,.39,.42),'full'),
    31:('clay',2.4,(1.,1.,1.),'full'),
    32:('aged_clay',2.4,(1.,1.,1.),'full'),
    33:('bitumen',2.,(1.15,1.15,1.15),'full'),
    34:('bitumen',2.,(.7,.74,.78),'full'),
    35:('granite',1.6,(.95,.87,.75),'full'),
}
# Do not impose the old procedural brick grid on photographic mortar.
# Keep stochastic blending for unstructured plaster/concrete, not peeling paint.
AUTHORED_MASK = sum(1 << (i-16) for i in (16,17,18,24,29,31,32,33,34,35))

def linear(a):
    a=np.asarray(a,np.float32)
    return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)

def srgb(a):
    a=np.clip(a,0,1)
    return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)

def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def read_map(z,suffix,optional=False):
    hits=[n for n in z.namelist() if n.endswith(suffix)]
    if not hits and optional:return None
    if len(hits)!=1:raise ValueError((z.filename,suffix,hits))
    data=z.read(hits[0])
    if hits[0].endswith('.exr'):
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
        import cv2
        a=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)
        if a is None:raise ValueError(hits[0])
        if a.ndim==3:a=a[:,:,[2,1,0]]
        a=np.asarray(a,np.float32)
    else:
        with Image.open(io.BytesIO(data)) as im:
            a=np.asarray(im.convert('RGB'),np.float32)/255
    if not np.isfinite(a).all():raise ValueError(hits[0])
    return a

def square_resize(a,size):
    # The array contract is square. Tile rectangular input rather than stretch.
    h,w=a.shape[:2]
    if w!=h:
        if w>h and w%h==0 and w//h<=4:
            a=np.concatenate([a]*(w//h),axis=0)
        elif h>w and h%w==0 and h//w<=4:
            a=np.concatenate([a]*(h//w),axis=1)
        else:raise ValueError(('unsupported aspect',h,w))
    if a.ndim==2:a=a[:,:,None]
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[:,:,i],np.float32)).resize((size,size),Image.Resampling.BOX),np.float32) for i in range(a.shape[2])],axis=-1)

def unit_normal(a):
    length=np.linalg.norm(a,axis=-1,keepdims=True)
    return np.where(length>1e-6,a/np.maximum(length,1e-6),np.array([0,0,1],np.float32))

def scalar(a):
    return a if a.ndim==2 else a[:,:,0]

def load_archive(path,size=512):
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist())>512*1024**2:raise ValueError('oversize archive')
        is_arm=path.name.endswith('_2k.zip')
        info={}
        if is_arm:
            a=read_map(z,'_diff_2k.jpg');n=read_map(z,'_nor_gl_2k.exr')
            orm=read_map(z,'_arm_2k.jpg');ao=scalar(read_map(z,'_ao_2k.jpg'))
            rough=scalar(read_map(z,'_rough_2k.exr'))
            metal=read_map(z,'_metal_2k.exr',True)
            # Use original separate linear maps where available; ARM is audited.
            info['arm_ao_mae']=float(np.abs(orm[:,:,0]-ao).mean())
            info['arm_roughness_mae']=float(np.abs(orm[:,:,1]-rough).mean())
            if max(info['arm_ao_mae'],info['arm_roughness_mae'])>.04:raise ValueError(('ARM channel mismatch',path.name,info))
            metal=scalar(metal) if metal is not None else np.zeros_like(rough)
            gltf=[x for x in z.namelist() if x.endswith('.gltf')]
            mat=json.loads(z.read(gltf[0]))['materials'][0] if len(gltf)==1 else {}
            scale=float(mat.get('normalTexture',{}).get('scale',1))
            metal*=float(mat.get('pbrMetallicRoughness',{}).get('metallicFactor',1))
            opacity=None
        else:
            a=read_map(z,'_Color.png');n=read_map(z,'_NormalGL.png')
            rough=scalar(read_map(z,'_Roughness.png'));ao=read_map(z,'_AmbientOcclusion.png',True)
            metal=read_map(z,'_Metalness.png',True);opacity=read_map(z,'_Opacity.png',True)
            ao=scalar(ao) if ao is not None else np.ones_like(rough)
            metal=scalar(metal) if metal is not None else np.zeros_like(rough)
            scale=1.
        h,w=a.shape[:2]
        if any(x.shape[:2]!=(h,w) for x in (n,rough,ao,metal)):raise ValueError(('unaligned maps',path.name))
        a=np.clip(square_resize(linear(a),size),0,1)
        n=square_resize(n*2-1,size);n[:,:,:2]*=scale
        n=unit_normal(n)*.5+.5
        orm=np.clip(square_resize(np.stack((ao,rough,metal),-1),size),0,1)
        maps=tuple(np.asarray(x,np.float32) for x in (a,n,orm))
        if any(x.shape!=(size,size,3) or not np.isfinite(x).all() for x in maps):raise ValueError(path.name)
        info.update(source=path.name,sha256=sha(path),source_size=[w,h],size=size,
                    normal_convention='OpenGL +Y',normal_scale=scale,
                    channels=['linearized base color','encoded tangent normal','AO roughness metalness'],
                    opacity_coverage=None if opacity is None else float((scalar(opacity)>.5).mean()),
                    displacement_present=any('displacement' in x.lower() or '_disp_' in x for x in z.namelist()),
                    fallback_ao=not is_arm and not any(x.endswith('_AmbientOcclusion.png') for x in z.namelist()),
                    fallback_metal=not is_arm and not any(x.endswith('_Metalness.png') for x in z.namelist()))
    return maps,info

def save_maps(root,maps):
    root.mkdir(parents=True,exist_ok=True)
    for k,m in zip(('basecolor','normal','orm'),maps):
        if k=='basecolor':m=srgb(m)
        Image.fromarray(np.uint8(np.rint(np.clip(m,0,1)*255))).save(root/(k+'.png'))

def load_all(root,out,size=512):
    if os.getenv('GITHUB_ACTIONS')=='true':raise RuntimeError('private source data must not enter CI')
    data={};manifest={}
    for name,file in SOURCES.items():
        data[name],manifest[name]=load_archive(root/file,size)
        save_maps(out/'normalized'/name,data[name])
    dump(out/'source_manifest.json',manifest)
    return data,manifest

def apply(arrays,metadata,data,manifest,out):
    if len(arrays[0])!=36 or len(arrays[1])!=36 or len(arrays[2])!=36:raise ValueError('expected stage-28 bank')
    meta=metadata.copy();rows=[]
    for i,(name,repeat,tint,mode) in SLOTS.items():
        a,n,r=(x.copy() for x in data[name])
        if mode=='hybrid':
            # Preserve the board geometry illusion (grooves and existing color).
            lum=a@np.array([.2126,.7152,.0722],np.float32)
            factor=np.clip(lum/max(float(np.median(lum)),.001),.65,1.35)
            a=arrays[0][i,:,:,:3]*(.70+.30*factor[:,:,None])
            old=arrays[1][i]*2-1;detail=n*2-1
            old[:,:,:2]+=detail[:,:,:2]*.25
            n=unit_normal(old)*.5+.5
            r=.70*arrays[2][i]+.30*r
            r[:,:,2]=0
        else:a=np.clip(a*np.asarray(tint,np.float32),0,1)
        maps=(a,n,r)
        # Quantize once in the actual storage conventions, matching GPU inputs.
        save_maps(out/'material_bank'/str(i),maps)
        q=[np.asarray(Image.open(out/'material_bank'/str(i)/(k+'.png')).convert('RGB'),np.float32)/255 for k in ('basecolor','normal','orm')]
        arrays[0][i,:,:,:3]=linear(q[0]);arrays[0][i,:,:,3]=1
        arrays[1][i]=q[1];arrays[2][i]=q[2]
        mask=np.isin(meta[:,0,0],[0,1,6,7])&(meta[:,0,1]==i)
        meta[mask,0,2]=repeat
        rows.append(dict(layer=i,source=name,mode=mode,repeat_m=repeat,records=int(mask.sum()),source_sha256=manifest[name]['sha256']))
    keep=np.ones(metadata.shape,bool)
    changed=np.isin(metadata[:,0,0],[0,1,6,7])&np.isin(metadata[:,0,1],list(SLOTS))
    keep[changed,0,2]=False
    np.testing.assert_array_equal(meta[keep],metadata[keep])
    dump(out/'overlay_manifest.json',rows)
    return meta,rows
