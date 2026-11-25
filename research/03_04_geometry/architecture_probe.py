"""Research-only facade layout compiler; rectangular shell, opaque colored proxy elements.
No real apertures, textures, CGAL, UNIGINE API, or surveyed architecture. Z-up core, Y-up GLB.
Dependencies: numpy, trimesh, shapely>=2.1 (geometry_probe.py from stage 03).
"""
import argparse, hashlib, json, math, random, time, sys
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import box
from geometry_probe import mesh_build

ROOT=Path(__file__).resolve().parent

def unit(*key):
    # Canonical stable substream; independent of evaluation/LOD order.
    return int.from_bytes(hashlib.blake2b(json.dumps(key,separators=(',',':')).encode(),digest_size=8).digest(),'big')/2**64

def fit_bays(length,p):
    avail=length-2*p['margin']; lo,hi=p['bay_range']
    if avail<=0:return []
    ns=range(max(1,math.ceil(avail/hi)),math.floor(avail/lo)+1)
    n=min(ns,key=lambda n:(abs(avail/n-p['bay_target']),n),default=0)
    if n==0:return [] # blind facade, do not squeeze windows
    return [(p['margin']+i*avail/n,avail/n) for i in range(n)]

def plan(building, p, w,d,n):
    fh=p['floor_height']; out=[]
    for side,L in enumerate((w,d,w,d)):
        bays=fit_bays(L,p)
        for i,(start,bw) in enumerate(bays):
            # Keep balcony stacks aligned. Ground level is never a balcony.
            stack=unit(building,'balcony_stack',side,i)<p['balcony_probability']*(1 if side==0 else .45)
            for f in range(n):
                door=(side==0 and f==0 and i==len(bays)//2)
                balcony=bool(stack and f>0 and not door)
                ww=min(p['window_width'],bw-.32) if not door else min(2.7 if p['roof']=='gable' and p['floor_height']>5 else 1.2,bw-.32)
                wh=min(p['window_height'],fh-p['sill']-.2) if not door else min(3.4 if p['floor_height']>5 else 2.2,fh-.2)
                z=f*fh+(0 if door else (.16 if balcony else p['sill']))
                if balcony: wh=min(2.15,fh-.36)
                out.append(dict(side=side,bay=i,floor=f,u=start+(bw-ww)/2,z=z,w=ww,h=wh,kind='door' if door else ('balcony_door' if balcony else 'window'),balcony=balcony,bay_width=bw,span=L))
    return out

def validate(layout,p,n):
    errors=[]; fh=p['floor_height']; by={}
    for a in layout:
        if not (a['w']>0 and a['h']>0 and a['u']>=p['margin']-1e-9 and a['u']+a['w']<=a['span']-p['margin']+1e-9 and a['z']>=a['floor']*fh-1e-9 and a['z']+a['h']<=(a['floor']+1)*fh+1e-9):errors.append('bounds')
        if a['balcony'] and (a['floor']==0 or a['kind']=='door'):errors.append('balcony_rule')
        by.setdefault((a['side'],a['floor']),[]).append(a)
    for aa in by.values():
        aa=sorted(aa,key=lambda a:a['u'])
        if any(x['u']+x['w']>y['u']+1e-9 for x,y in zip(aa,aa[1:])):errors.append('overlap')
    return errors

def box_mesh(center,extent,color,transform=None):
    m=trimesh.creation.box(extents=extent);m.apply_translation(center)
    if transform is not None:m.apply_transform(transform)
    m.visual.vertex_colors=np.tile([*color,255],(len(m.vertices),1));return m

def emit(building,p,w,d,n,layout,lod):
    fh=p['floor_height']; H=fh*n; shell,_=mesh_build(box(0,0,w,d),p['roof'],H)
    # shell face color from stage 03: replace exact palette
    color=shell.visual.face_colors.copy(); roof=(color[:,0]==153); wall=(color[:,0]==219)
    tint=round((unit(building,'tint')-.5)*18)
    wc=[max(0,min(255,c+tint)) for c in p['wall']]
    color[roof]=[*p['roof_color'],255];color[wall]=[*wc,255];shell.visual.face_colors=color
    shell.unmerge_vertices();shell.visual.vertex_colors=np.repeat(color,3,axis=0)
    meshes=[shell]
    # local wall u forward, v outward, z up; reflected basis? choose outward right, so basis det -1.
    frames=[((0,0),(1,0),(0,-1)),((w,0),(0,1),(1,0)),((w,d),(-1,0),(0,1)),((0,d),(0,-1),(-1,0))]
    def part(side,u,v,z,du,dv,dz,c):
        o,t,q=frames[side];T=np.eye(4);T[:2,0]=t;T[:2,1]=q;T[:2,3]=o
        meshes.append(box_mesh((u,v,z),(du,dv,dz),c,T))
    for side,L in enumerate((w,d,w,d)):
        doors=[a for a in layout if a['side']==side and a['kind']=='door']
        intervals=[(0,L)] if not doors else [(0,doors[0]['u']),(doors[0]['u']+doors[0]['w'],L)]
        for lo,hi in intervals:
            if hi-lo>1e-6:part(side,(lo+hi)/2,.035,.18,hi-lo,.07,.36,p['accent'])
        if lod==0 and building.split(':')[0] in ('panel_block','modern_apartment','resort_hotel'):
            for f in range(1,n):part(side,L/2,.018,f*fh,L,.036,.055,p['accent'])
    for a in layout:
        s,u,z,ww,hh=a['side'],a['u'],a['z'],a['w'],a['h']; mid=u+ww/2; zz=z+hh/2
        dark=[61,81,91] if unit(building,'glass',s,a['bay'],a['floor'])>.2 else [103,123,131]
        if a['kind']=='door':dark=[94,91,85]
        if lod==0:
            part(s,mid,.06,zz,ww,.12,hh,[229,226,215])
            part(s,mid,.128,zz,ww-.12,.018,hh-.12,dark)
            if a['kind']!='door':part(s,mid,.145,zz,.035,.025,hh-.12,[221,219,212])
            if a['balcony']:
                depth=p['balcony_depth']; bw=min(a['bay_width']-.16,ww+.32); slab_z=a['floor']*fh+.08
                part(s,mid,depth/2,slab_z,bw,depth,.16,p['accent'])
                part(s,mid,depth-.045,slab_z+.55,bw,.07,.9,[139,151,154] if building.startswith(('modern','resort')) else [125,120,113])
                for sign in [-1,1]:part(s,mid+sign*(bw/2-.035),depth/2,slab_z+.55,.07,depth,.9,[139,151,154])
        else:
            part(s,mid,.012,zz,ww,.024,hh,dark) # geometric proxy for eventual baked/UV facade
    if p['roof']=='flat':
        for s,L in enumerate((w,d,w,d)):part(s,L/2,-.06,H+.18,L,.18,.36,p['accent'])
    return trimesh.util.concatenate(meshes)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profiles',type=Path,default=ROOT/'blacksea_profiles.json');ap.add_argument('--out',type=Path,default=ROOT);a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    profiles=json.loads(a.profiles.read_text())['profiles']; rng=random.Random(46)
    report={'scope':'synthetic rectangles only; logical slot tests not full mesh collision checks','cases':0,'violations':0,'empty_facades':0,'repeat_order_mismatches':0,'samples':[]}
    start=time.perf_counter()
    for name,p in profiles.items():
        for i in range(200):
            w,d=[rng.uniform(1.0,65) for _ in range(2)];n=rng.randint(*p['floor_range']);bid=f'{name}:{i}'
            lay=plan(bid,p,w,d,n);report['cases']+=1;report['violations']+=len(validate(lay,p,n));report['empty_facades']+=4-len(set(x['side'] for x in lay))
            report['repeat_order_mismatches']+=int(lay!=plan(bid,p,w,d,n))
    report['layout_test_seconds']=round(time.perf_counter()-start,4)
    scene=trimesh.Scene(); coarse=trimesh.Scene(); previews=[]
    for row,(name,p) in enumerate(profiles.items()):
        for j in range(3):
            w,d=p['sample_size']; w*=.85+.15*j;d*=1+.08*j;n=max(p['floor_range'][0],min(p['floor_range'][1],p['sample_floors']+j-1));bid=f'{name}:{1000+j}'
            lay=plan(bid,p,w,d,n); assert not validate(lay,p,n)
            m0=emit(bid,p,w,d,n,lay,0);m1=emit(bid,p,w,d,n,lay,1)
            report['samples'].append(dict(id=bid,floors=n,windows=sum(x['kind']=='window' for x in lay),balconies=sum(x['balcony'] for x in lay),lod0_triangles=len(m0.faces),lod1_proxy_triangles=len(m1.faces),layout_hash=hashlib.sha256(json.dumps(lay,sort_keys=True).encode()).hexdigest()))
            previews.append((name,j,m0.copy()))
            for sc,m in ((scene,m0),(coarse,m1)):
                m.apply_translation([j*55,row*34,0]);sc.add_geometry(m,geom_name=bid,node_name=bid)
    for sc,fn in [(scene,'architecture_probe_lod0.glb'),(coarse,'architecture_probe_lod1.glb')]:
        sc.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2,[1,0,0]));path=a.out/fn;path.write_bytes(sc.export(file_type='glb'));assert len(trimesh.load(path,force='scene').geometry)==18
    report['totals']={k:sum(x[k] for x in report['samples']) for k in ('windows','balconies','lod0_triangles','lod1_proxy_triangles')}
    (a.out/'architecture_probe_results.json').write_text(json.dumps(report,indent=2))
    # Orthographic contact sheet from actual geometry; no synthetic image generation.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig,axes=plt.subplots(6,3,figsize=(12,15),facecolor='#f5f3ee')
    view=np.array([1.,-1.5,1.05]);view/=np.linalg.norm(view);right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
    for (name,j,m),ax in zip(previews,axes.flat):
        vs=m.vertices;ts=m.faces;xy=np.column_stack((vs@right,vs@up));depth=vs@view
        colors=m.visual.vertex_colors[ts].mean(1)[:,:3]/255.; light=np.clip(.55+.45*(m.face_normals@np.array([.3,-.5,.81])),.38,1.0);colors*=light[:,None]
        # Simple depth buffer; avoids painter-order artifacts on long wall triangles.
        size=400; lo=xy.min(0);hi=xy.max(0);scale=(size-28)/max(hi-lo);q=(xy-(hi+lo)/2)*scale+size/2
        buf=np.full((size,size),-np.inf); canvas=np.full((size,size,3),[245,243,238],dtype=np.uint8)
        for k,(i0,i1,i2) in enumerate(ts):
            a0,b0,c0=q[[i0,i1,i2]]; mn=np.maximum(0,np.floor(np.min([a0,b0,c0],axis=0)).astype(int));mx=np.minimum(size-1,np.ceil(np.max([a0,b0,c0],axis=0)).astype(int))
            if np.any(mx<mn):continue
            ab=b0-a0;ac=c0-a0;den=ab[0]*ac[1]-ab[1]*ac[0]
            if abs(den)<1e-9:continue
            xx,yy=np.meshgrid(np.arange(mn[0],mx[0]+1)+.5,np.arange(mn[1],mx[1]+1)+.5);dx=xx-a0[0];dy=yy-a0[1]
            u=(dx*ac[1]-dy*ac[0])/den;v=(ab[0]*dy-ab[1]*dx)/den
            z=depth[i0]+u*(depth[i1]-depth[i0])+v*(depth[i2]-depth[i0]);sub=buf[mn[1]:mx[1]+1,mn[0]:mx[0]+1]; mask=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(z>sub)
            sub[mask]=z[mask];canvas[mn[1]:mx[1]+1,mn[0]:mx[0]+1][mask]=(np.clip(colors[k],0,1)*255).astype(np.uint8)
        ax.imshow(canvas,origin='lower');ax.axis('off');ax.set_title(f'{name.replace("_"," ")} / {j+1}',fontsize=10)
    fig.suptitle('Black Sea architecture — procedural geometry probe',fontsize=18,y=.99);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(a.out/'architecture_probe_contact.png',dpi=130);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
