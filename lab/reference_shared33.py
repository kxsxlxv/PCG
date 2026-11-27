"""Shared measured facade patches on new canonical GIS polygons.
One atlas / one material for all generated meshes. Diagnostic flat-roof lane;
no donor mesh or full facade is reused. Source-derived strips are hypotheses.
"""
import argparse,io,json,hashlib,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pcg.footprint import Footprint
from pcg.massing import build_massing
from reference_intake33 import Gltf,dump
from reference_pbr33 import BakeRenderer,glb_write

def image_arrays(path):
    g=Gltf(path);out=[]
    for im in g.doc['images']:
        assert 'bufferView' in im
        out.append(np.asarray(Image.open(io.BytesIO(g.view(im['bufferView']))).convert('RGB')).copy())
    return out

def groups(mask,gap=5):
    x=np.flatnonzero(mask)
    if not len(x):return []
    return [(int(a[0]),int(a[-1]+1)) for a in np.split(x,np.flatnonzero(np.diff(x)>gap)+1)]

def patch_regions(rough,cuts,aspect):
    h,w=rough.shape
    out=[]
    for y,Y in zip(cuts[:-1],cuts[1:]):
        if not (0<=y<Y<=h):continue
        dark=rough[y:Y]<32
        if dark.mean()<.015:continue
        spans=groups(dark.mean(axis=0)>.12,max(3,w//180))
        for i,(a,b) in enumerate(spans):
            left=(spans[i-1][1]+a)//2 if i else max(0,a-int((b-a)*.32))
            right=(b+spans[i+1][0])//2 if i+1<len(spans) else min(w,b+int((b-a)*.32))
            ratio=(right-left)/(Y-y)*aspect
            if left<.025*w or right>.975*w or not .4<=ratio<=3.5:continue
            f=float(dark[:,left:right].mean())
            if .055<=f<=.65 and right-left>=12:out.append((left,y,right,Y,ratio,f))
    return out

def signature(rough):
    dark=rough<32
    ys,xs=np.where(dark)
    if not len(xs):return dict(glass_span=0.,glass_center=.5,glass_height=0.,glass_class='none')
    w=rough.shape[1];h=rough.shape[0]
    width=(np.quantile(xs,.95)-np.quantile(xs,.05)+1)/w
    height=(np.quantile(ys,.95)-np.quantile(ys,.05)+1)/h
    return dict(glass_span=float(width),glass_center=float(np.median(ys)/h),glass_height=float(height),glass_class='wide' if width>=.63 else 'narrow')

def compatible(a,b):
    # Roughness-derived shape compatibility only, not a semantic balcony ID.
    return a.get('glass_class')==b.get('glass_class') and abs(a.get('glass_center',.5)-b.get('glass_center',.5))<=.12 and abs(a.get('glass_span',0)-b.get('glass_span',0))<=.20 and abs(a.get('glass_height',0)-b.get('glass_height',0))<=.23 and .75<=a['ratio']/b['ratio']<=1.33

def authored_ground_modules(plain):
    # Explicit authored entry/plinth, with wall color drawn from the same
    # source family. No assertion these are actual donor doors.
    size=128
    p=[np.asarray(Image.fromarray(a).resize((size,size),Image.Resampling.BILINEAR)).copy() for a in plain]
    pl=[a.copy() for a in p];pl[0]=(pl[0].astype(float)*.65).astype(np.uint8)
    entry=[a.copy() for a in p]
    entry[0][106:]=(entry[0][106:].astype(float)*.65).astype(np.uint8)
    entry[0][24:,44:84]=[70,65,58];entry[0][28:,48:80]=[91,70,53]
    entry[0][31:57,51:77]=[47,60,65];entry[0][30:58,63:65]=[112,108,97]
    entry[0][82:85,73:78]=[165,159,139];entry[0][23:27,40:88]=[101,98,90]
    entry[1][24:,44:84]=[128,128,255];entry[2][24:,44:84]=[255,160,0]
    entry[2][31:57,51:77,1]=35
    return pl,entry

def extract_modules(pbr,reference):
    models={r['id']:r for r in reference['models']}
    selected=('high_001','high_021','high_024','high_029')
    modules=[];families={}
    for key in selected:
        a=image_arrays(pbr/(key+'_pbr.glb'));meta=json.loads((pbr/(key+'_facades.json')).read_text())
        size=a[0].shape[0]//2;span=np.array(models[key]['span_source_units'])
        assert a[0].shape[:2]==(size*2,size*3)
        def tile(i):
            y,x=divmod(i,3);return [im[y*size:(y+1)*size,x*size:(x+1)*size] for im in a]
        choices=[]
        for side in meta['sides'][:4]:
            i=side['side'];strip=side['floor_strip']
            if strip is None:continue
            tt=tile(i);ratio=span[1 if i<2 else 0]/span[2]
            for x,y,X,Y,r,f in patch_regions(tt[2][:,:,1],strip['cuts'],ratio):
                choices.append(dict(side=i,rect=[x,y,X,Y],ratio=r,glass_fraction=f))
        if not choices:continue
        # Coherent width family; choose diverse source positions within this width band.
        med=float(np.median([q['ratio'] for q in choices]))
        choices=[q for q in choices if .75*med<=q['ratio']<=1.35*med]
        choices.sort(key=lambda q:(q['side'],q['rect'][1],q['rect'][0]))
        if len(choices)>12:choices=[choices[i] for i in np.unique(np.linspace(0,len(choices)-1,12).round().astype(int))]
        ids=[]
        for q in choices:
            x,y,X,Y=q['rect'];tt=tile(q['side']);ids.append(len(modules));patch=[im[y:Y,x:X].copy() for im in tt];modules.append((dict(family=key,kind='glazing_patch',**q,**signature(patch[2][:,:,1])),patch))
        # Plain wall module: find a genuinely low-glass crop in the same facade.
        tt=tile(choices[0]['side']);rough=tt[2][:,:,1]
        y=choices[0]['rect'][1];Y=choices[0]['rect'][3];w=max(20,min(size//5,Y-y))
        candidates=[]
        for x in range(size//12,size-size//12-w,max(4,w//4)):
            f=float((rough[y:Y,x:x+w]<32).mean())
            candidates.append((f,x))
        f,x=min(candidates)
        if f<=.015:
            plain=[im[y:Y,x:x+w].copy() for im in tt];plain_origin=[x,y,x+w,Y]
        else:
            # Explicit neutral fallback for blind walls; never paste visible windows.
            al=tt[0][y:Y];rr=rough[y:Y];valid=rr>=32
            c=np.median(al[valid],axis=0).astype(np.uint8) if valid.any() else np.array([140,140,130],np.uint8)
            plain=[np.broadcast_to(v,(32,32,3)).copy() for v in (c,np.array([128,128,255],np.uint8),np.array([255,180,0],np.uint8))];plain_origin=None
        wall=len(modules);modules.append((dict(family=key,kind='blind_wall',side=choices[0]['side'],rect=plain_origin,ratio=1.,fallback=plain_origin is None),plain))
        tt=tile(4);a0=size*3//8;b0=size*5//8
        roof=len(modules);modules.append((dict(family=key,kind='roof',side=4,rect=[a0,a0,b0,b0],ratio=1.),[im[a0:b0,a0:b0].copy() for im in tt]))
        pl,entry=authored_ground_modules(plain)
        plinth=len(modules);modules.append((dict(family=key,kind='plinth',ratio=1.,authored=True),pl))
        door=len(modules);modules.append((dict(family=key,kind='entry',ratio=1.,authored=True),entry))
        families[key]=dict(glazing=ids,blind=wall,roof=roof,plinth=plinth,entry=door,ratio=float(np.median([modules[i][0]['ratio'] for i in ids])))
    return modules,families

def atlas_pack(modules,cell=256,gutter=4):
    cols=8;rows=math.ceil(len(modules)/cols);inner=cell-2*gutter
    atlas=[np.zeros((rows*cell,cols*cell,3),np.uint8) for _ in range(3)];catalog=[]
    for i,(meta,maps) in enumerate(modules):
        y,x=divmod(i,cols)
        for dst,im in zip(atlas,maps):
            a=np.asarray(Image.fromarray(im).resize((inner,inner),Image.Resampling.LANCZOS))
            a=np.pad(a,((gutter,gutter),(gutter,gutter),(0,0)),mode='edge')
            dst[y*cell:(y+1)*cell,x*cell:(x+1)*cell]=a
        # Half-texel inset at the content edge, plus duplicated gutter.
        W,H=cols*cell,rows*cell
        catalog.append(dict(id=i,**meta,uv=[(x*cell+gutter+.5)/W,(y*cell+gutter+.5)/H,((x+1)*cell-gutter-.5)/W,((y+1)*cell-gutter-.5)/H]))
    return atlas,catalog

def stable(*args):return int.from_bytes(hashlib.sha256('|'.join(map(str,args)).encode()).digest()[:8],'little')
def face_uv(rect,uv):
    a,b,c,d=rect;return np.asarray(uv)*[c-a,d-b]+[a,b]

def build(f,floors,height,family,catalog,party=(),cap=1800):
    assert isinstance(floors,int) and floors>0 and height>0
    mass=build_massing(f,height).shell;verts=np.array(mass.vertices);rings=(f.exterior,*f.holes)
    h=height/floors;edges=[(r,i,np.array(a),np.array(b)) for r,ring in enumerate(rings) for i,(a,b) in enumerate(zip(ring,ring[1:]+ring[:1]))]
    base=min(.55,h*.18);crown=min(.24,h*.08);fh=(height-base-crown)/floors
    ncap=sum(len(s.vertices)-2 for s in mass.surfaces if s.role!='wall')
    bays=[1 if (r,i) in party else max(1,round(np.linalg.norm(b-a)/(fh*family['ratio']))) for r,i,a,b in edges]
    def cost():return ncap+2*sum(3 if (r,i) in party else (floors+2)*n for (r,i,a,b),n in zip(edges,bays))
    while cost()>cap:
        idx=max(range(len(bays)),key=lambda i:bays[i])
        if bays[idx]<=1:raise ValueError('budget cannot fit one bay per floor')
        bays[idx]-=1
    t=[];u=[];used=[];wall_area=0.;stacks=0;entry_count=0;variation_count=0
    main=max(((np.linalg.norm(b-a),i) for r,i,a,b in edges if r==0 and (r,i) not in party),default=(0,-1))[1]
    def add(q,uv,mid):
        t.append(np.asarray(q));u.append(face_uv(catalog[mid]['uv'],uv));used.append(mid)
    def quad(A,B,z0,z1,mid):
        nonlocal wall_area
        if z1-z0<=1e-10:return
        q=np.array([[*A,z0],[*B,z0],[*B,z1],[*A,z1]])
        uv=np.array([[0,1],[1,1],[1,0],[0,0]])
        for ids in ([0,1,2],[0,2,3]):add(q[ids],uv[ids],mid)
        wall_area+=np.linalg.norm(B-A)*(z1-z0)
    for (r,i,a,b),nb in zip(edges,bays):
        blind=(r,i) in party
        if blind:
            quad(a,b,0,base,family.get('plinth',family['blind']))
            quad(a,b,base,height-crown,family['blind'])
            quad(a,b,height-crown,height,family['blind'])
            continue
        length=float(np.linalg.norm(b-a))
        # Entrances placed on the longest available exterior face only.
        entry_bays=set()
        if r==0 and i==main and length>=8 and nb>=3 and 'entry' in family:
            sections=max(1,round(length/19.))
            entry_bays={min(nb-2,max(1,int((j+.5)*nb/sections))) for j in range(sections)}
        for j in range(nb):
            A=a+(b-a)*j/nb;B=a+(b-a)*(j+1)/nb
            ids=family['glazing']
            anchor=ids[stable(f.building_id,r,i,j,'stack')%len(ids)]
            pool=[k for k in ids if compatible(catalog[anchor],catalog[k])] or [anchor]
            stacks+=1
            if j not in entry_bays:quad(A,B,0,base,family.get('plinth',family['blind']))
            for z in range(floors):
                z0=base+fh*z;z1=base+fh*(z+1)
                if z==0 and j in entry_bays:
                    quad(A,B,0,z1,family['entry']);entry_count+=1
                else:
                    mid=pool[stable(f.building_id,r,i,z,j,'finish')%len(pool)]
                    variation_count+=int(mid!=anchor)
                    quad(A,B,z0,z1,mid)
            quad(A,B,height-crown,height,family['blind'])
    lo=verts[:,:2].min(0);span=np.ptp(verts[:,:2],axis=0);roof_area=0.
    for s in mass.surfaces:
        if s.role=='wall':continue
        p=verts[list(s.vertices)];uv=(p[:,:2]-lo)/span;uv[:,1]=1-uv[:,1]
        mid=family['roof'] if s.role=='roof' else family['blind']
        for j in range(1,len(p)-1):
            ids=[0,j,j+1];add(p[ids],uv[ids],mid)
            if s.role=='roof':roof_area+=abs(np.cross(p[j]-p[0],p[j+1]-p[0])[2])*.5
    T=np.array(t);U=np.array(u);assert len(T)<=cap
    assert np.isclose(roof_area,f.polygon().area,rtol=1e-10)
    assert np.isclose(wall_area,f.polygon().length*height,rtol=1e-10)
    assert np.isclose(T[:,:,2].min(),0) and np.isclose(T[:,:,2].max(),height)
    return T,U,dict(id=f.building_id,footprint_digest=f.digest(),height=height,floors=floors,triangles=len(T),roof_area=roof_area,wall_area=wall_area,modules=sorted(set(used)),party_edges=len(party),column_stacks=stacks,authored_entries=entry_count,compatible_variations=variation_count)

def fixtures():
    return [
        (Footprint('shared-slab',[(0,0),(46,0),(46,13),(0,13)]),5,15,()),
        (Footprint('shared-L',[(0,0),(30,0),(30,12),(15,12),(15,25),(0,25)]),7,21,()),
        (Footprint('shared-court',[(0,0),(35,0),(35,30),(0,30)], [[(9,8),(9,22),(26,22),(26,8)]]),6,18,()),
        (Footprint('shared-tower',[(0,0),(18,4),(14,22),(-4,18)]),14,42,()),
        (Footprint('shared-party',[(0,0),(18,0),(18,11),(0,11)]),8,24,((0,1),)),
    ]

def run(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    ref=json.loads((args.reference/'report.json').read_text())
    mods,families=extract_modules(args.pbr,ref)
    assert len(families)>=3 and len(mods)>=12
    atlas,cat=atlas_pack(mods);dump(out/'modules.json',dict(families=families,modules=cat))
    for name,a in zip(('albedo','normal','orm'),atlas):Image.fromarray(a).save(out/(name+'.png'))
    render=BakeRenderer();render.maps(atlas);rows=[];ts=[];us=[]
    keys=list(families)
    for i,(f,floors,h,party) in enumerate(fixtures()):
        family=families[keys[i%len(keys)]]
        t,u,r=build(f,floors,h,family,cat,party)
        tt,uu,rr=build(f,floors,h,family,cat,party)
        np.testing.assert_array_equal(t,tt);np.testing.assert_array_equal(u,uu);assert r==rr
        r['family']=keys[i%len(keys)];rows.append(r)
        span=np.ptp(t.reshape(-1,3),axis=0);center=(t.reshape(-1,3).max(0)+t.reshape(-1,3).min(0))/2;ext=np.linalg.norm(span)*.55
        im=render.draw(t,u,center,(ext*4/3,ext),(1,-1.7,.65),1,(1280,960))[0]
        Image.fromarray(im).save(out/(f.building_id+'.png'))
        ts.append(t+np.array([(i%3)*65.,(i//3)*55.,0.]));us.append(u)
    t=np.concatenate(ts);u=np.concatenate(us)
    t,u=glb_write(out/'shared_city.glb',t,u,atlas)
    center=(t.reshape(-1,3).max(0)+t.reshape(-1,3).min(0))/2;ext=np.linalg.norm(np.ptp(t.reshape(-1,3),axis=0))*.55
    im=render.draw(t,u,center,(ext*16/9,ext),(1,-1.7,.85),1,(1600,900))[0];Image.fromarray(im).save(out/'shared_city.png')
    stress=[]
    for i in range(128):
        w=12+i%40;d=10+(i*7)%23;floors=3+i%12
        f=Footprint('stress-'+str(i),[(0,0),(w,0),(w,d),(0,d)])
        a,b,r=build(f,floors,3.*floors,families[keys[i%len(keys)]],cat)
        stress.append(r['triangles'])
    result=dict(families=len(families),modules=len(mods),glazing_modules=sum(m['kind']=='glazing_patch' for m in cat),atlas_size=[atlas[0].shape[1],atlas[0].shape[0]],buildings=rows,stress_cases=len(stress),stress_max_triangles=max(stress),single_material=True,limitations=['Shared floor/bay patches use roughness shape compatibility, not certified semantic window or balcony classes.','Flat roof research shell only; entry and plinth textures authored from family colors, no Stage32 balcony geometry or roof equipment integration.','Roof UVs stretch across entire footprint; texture scale not certified.','No claim of native UNIGINE runtime performance.','Exact geometry on synthetic footprints, not live GIS data.'])
    dump(out/'report.json',result);print('SHARED33_RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('reference','pbr','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
