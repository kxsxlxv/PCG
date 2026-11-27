"""Opt-in Stage33 material overlay on the actual accepted Stage32 exports.
No host mesh, UV, PCGF data, cutout or original opening record is edited.
The fourth/fifth opening texels are a research GPU sidecar, not native ABI.
"""
from pathlib import Path
import argparse,hashlib,json,math,shutil
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False))
def verify(root):
    m=json.loads((root/'bundle.json').read_text())
    for name,d in m.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==d,name
    return len(m)
def stable(*v):return int.from_bytes(hashlib.sha256('|'.join(map(str,v)).encode()).digest()[:8],'little')
def crop_window(rough,ratio):
    # Roughness is a candidate glass mask, not a certified window classifier.
    y,x=np.where(rough<32)
    if len(x)<rough.size*.035:return None
    h,w=rough.shape
    x0=max(0,int(np.quantile(x,.03)-.055*w));x1=min(w,int(np.quantile(x,.97)+.055*w)+1)
    y0=max(0,int(np.quantile(y,.03)-.055*h));y1=min(h,int(np.quantile(y,.97)+.055*h)+1)
    if x1-x0<12 or y1-y0<12:return None
    a=ratio*((x1-x0)/w)/((y1-y0)/h)
    frac=float((rough[y0:y1,x0:x1]<32).mean())
    if not .25<=a<=5. or not .07<=frac<=.98:return None
    return (x0,y0,x1,y1),float(a),frac

def frame_palette(a,rough):
    # Candidate frame color: pixels close to low-roughness region, not the
    # brightest arbitrary facade pixel. Fallback is neutral.
    import cv2
    glass=(rough<32).astype(np.uint8)
    width=max(2,min(10,min(glass.shape)//16))
    ring=(cv2.dilate(glass,np.ones((width*2+1,width*2+1),np.uint8))>0)&(glass==0)&(rough>=45)
    if ring.sum()<16:return [.50,.50,.48]
    q=np.median(a[ring],axis=0)/255.
    q=np.where(q<=.04045,q/12.92,((q+.055)/1.055)**2.4)
    return np.clip(q,.10,.82).tolist()

def wall_detail(maps):
    import cv2
    # Remove low-frequency shading/edge imprint from projected wall crop.
    # Encode bounded high-pass residual around sRGB 0.5.
    a=maps[0].astype(np.float32)/255
    low=cv2.GaussianBlur(a,(0,0),max(3.,min(a.shape[:2])/12))
    a=np.clip(.5+(a-low)*.40,.44,.56)
    n=maps[1].astype(np.float32)/127.5-1.
    n[:,:,:2]=np.clip(n[:,:,:2],-.35,.35)
    n[:,:,2]=np.sqrt(np.maximum(.1,1.-(n[:,:,:2]**2).sum(2)))
    return [np.rint(a*255).astype(np.uint8),np.rint((n+1)*127.5).astype(np.uint8),maps[2]]

def bank(root):
    data=json.loads((root/'modules.json').read_text())
    imgs=[np.asarray(Image.open(root/(k+'.png')).convert('RGB')).copy() for k in ('albedo','normal','orm')]
    H,W=imgs[0].shape[:2];assert all(x.shape==imgs[0].shape for x in imgs)
    layers=[];catalog=[]
    for m in data['modules']:
        if m['kind'] not in ('glazing_patch','blind_wall'):continue
        x,y,X,Y=np.round(np.array(m['uv'])*[W,H,W,H]).astype(int)
        maps=[im[y:Y,x:X].copy() for im in imgs]
        if m['kind']=='glazing_patch':
            q=crop_window(maps[2][:,:,1],m['ratio'])
            if q is None:continue
            (a,b,A,B),aspect,frac=q;maps=[im[b:B,a:A] for im in maps]
        else:
            if m.get('fallback') or (maps[2][:,:,1]<32).mean()>.02:continue
            aspect=1.;frac=float((maps[2][:,:,1]<32).mean())
        palette=frame_palette(maps[0],maps[2][:,:,1]) if m['kind']=='glazing_patch' else None
        if m['kind']=='blind_wall':maps=wall_detail(maps)
        catalog.append(dict(index=len(layers),source_module=m['id'],family=m['family'],kind=m['kind'],aspect=aspect,low_roughness_fraction=frac,frame_rgb=palette))
        layers.append(maps)
    assert len(layers)>0 and len({x['family'] for x in catalog if x['kind']=='glazing_patch'})>=3
    return layers,catalog

def assignments(meta,ops,table,catalog,base):
    # Original 3 vec4s are bitwise identical. Appended slot IDs are 1-based,
    # zero is opt-out. Host aliases use exactly the same slot addresses.
    assert ops.ndim==3 and ops.shape[1:]==(3,4)
    ext=np.zeros((len(ops),5,4),np.float32);ext[:,:3]=ops
    families=sorted({x['family'] for x in catalog if x['kind']=='glazing_patch'})
    by={k:[x for x in catalog if x['family']==k and x['kind']=='glazing_patch'] for k in families}
    walls={k:next((x for x in catalog if x['family']==k and x['kind']=='blind_wall'),None) for k in families}
    seen=set();count=0;walls_count=0;buildings={}
    for r in sorted(table,key=lambda x:x['id']):
        j=r['id']
        if r.get('role')!='wall' or meta[j,0,0]!=1 or meta[j,8,0]>=1.5:continue
        start,bays,floors,_=map(int,meta[j,1])
        if not bays or not floors:continue
        bid=r['building'];key=families[stable(bid)%len(families)];pool=by[key]
        for z in range(floors):
            for bay in range(bays):
                slot=start+z*bays+bay
                assert slot not in seen;seen.add(slot)
                role,paired=ops[slot,2,:2]
                if walls[key] is not None:
                    ext[slot,3,1]=base+walls[key]['index']+1;walls_count+=1
                if role!=1 or paired!=0:continue
                q=ops[slot,0];w,h=q[2:]-q[:2]
                if min(w,h)<=0:continue
                a=float(w/h)
                candidates=sorted(pool,key=lambda x:(abs(math.log(a/x['aspect'])),x['index']))
                candidates=[x for x in candidates if abs(math.log(a/x['aspect']))<=math.log(2.0)][:3]
                if not candidates:continue
                # Stable vertical stack, restricted to geometrically compatible apertures.
                x=candidates[stable(bid,r.get('facade_id',j),bay)%len(candidates)]
                ext[slot,3,0]=base+x['index']+1;ext[slot,4,:3]=x.get('frame_rgb',[.50,.50,.48]);count+=1
                buildings[bid]=buildings.get(bid,0)+1
    np.testing.assert_array_equal(ext[:,:3],ops)
    selected=ext[:,3,0]>0
    assert np.all(ops[selected,2,0]==1) and np.all(ops[selected,2,1]==0)
    return ext,dict(windows=count,wall_slots=walls_count,buildings=buildings,layer_ids=sorted(set(int(x)-1 for x in ext[:,3,0] if x>0)))

def append_arrays(arrays,layers):
    import gpu_lab as gpu
    for j in range(3):
        size=arrays[j].shape[1]
        add=[]
        for maps in layers:
            a=np.asarray(Image.fromarray(maps[j]).resize((size,size),Image.Resampling.LANCZOS),np.float32)/255
            if j==0:a=np.dstack((gpu.srgb(a),np.ones(a.shape[:2],np.float32)))
            add.append(a)
        arrays[j]=np.concatenate((arrays[j],np.asarray(add,np.float32)))

def run(source,atlas,out):
    import zero_reference_grammar as ref
    import zero_photoreal as prev
    import gpu_lab as gpu
    out.mkdir(parents=True,exist_ok=False)
    source_files=verify(source);atlas_files=verify(atlas);verify(ref.ROOT)
    layers,cat=bank(atlas)
    arrays,mips,info=ref.load_materials(out,None)
    base=len(arrays[0]);assert base==36
    append_arrays(arrays,layers)
    for i,maps in enumerate(layers):
        p=out/'donor_bank'/str(base+i);p.mkdir(parents=True)
        for name,a in zip(('albedo','normal','orm'),maps):Image.fromarray(a).save(p/(name+'.png'))
    dump(out/'donor_layers.json',dict(base=base,layers=cat,source_atlas_manifest=atlas_files))
    old_report=json.loads((source/'report.json').read_text())
    rows=[];assignments_report=[];image_rows=[];closeups=[]
    for variant in (0,1):
        p=source/f'grammar_{variant}';dst=out/f'tile_{variant}';dst.mkdir()
        table=json.loads((p/'parameters.json').read_text());records=json.loads((p/'plans.json').read_text())
        placements=json.loads((p/'placements.json').read_text());bounds=prev.envelopes(records,placements)
        remap={(r['building'],r['kind']):r['id'] for r in json.loads((p/'cutout_records.json').read_text())}
        with np.load(p/'gpu_metadata.npz') as z:meta=z['metadata'];ops=z['openings'];proxy=z['reflection_proxies'];cut=z['cutout_face_records']
        def load(name):
            scene,owner,cuts,nv,_=prev.load_scene(p,name,table,bounds)
            scene,face=prev.remap_scene(scene,cuts,remap,nv);np.testing.assert_array_equal(face,cut)
            return scene,owner
        near,no=load('tile_opaque.glb');lite,lo=load('tile_lite_opaque.glb')
        ext,stats=assignments(meta,ops,table,cat,base)
        rev,stats2=assignments(meta,ops,list(reversed(table)),cat,base);np.testing.assert_array_equal(ext,rev);assert stats==stats2
        assignments_report.append(stats)
        np.savez_compressed(dst/'donor_opening_sidecar.npz',openings=ext)
        # Exact byte identity with the accepted Stage32 source.
        for name in ('tile_opaque.glb','tile_lite_opaque.glb','roof_balcony_cutout.glb','antenna_cutout.glb','facade_params.bin','facade_cells.bin','gpu_metadata.npz','parameters.json','plans.json','placements.json','cutout_records.json'):
            shutil.copyfile(p/name,dst/name);assert sha(dst/name)==sha(p/name)
        vn=gpu.vertex_data(near);vl=gpu.vertex_data(lite)
        renderer=gpu.GL(vn,meta,ext,arrays,photoreal=True,cutout_mips=mips,natural=True,compact_materials=True)
        renderer.tex2(proxy,9,'reflectionProxies',False)
        def draw(verts,center,ppm,view,mode,enable):
            renderer.uniform('q33Enabled',int(enable));renderer.replace_vertices(verts)
            pts=verts[:,:3];renderer.shadow_center=(pts.min(0)+pts.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(pts,axis=0))*1.2))
            im,_=renderer.render(center,ppm,view,mode,(1280,960),info)
            return im,renderer.last_mask.copy()
        for spec in ref.fixtures(variant):
            bid=spec['footprint'].building_id;key=spec['id'];poly=np.asarray(spec['footprint'].exterior);off=np.asarray(spec['offset'])
            a=np.r_[poly.min(0),0]+off;b=np.r_[poly.max(0),spec['top']]+off
            center=(a+b)/2;camera=(-1.,1.6,.8) if spec['party'] else (1.,-1.6,.8)
            box=np.array([[x,y,z] for x in (a[0]-2.5,b[0]+2.5) for y in (a[1]-2.5,b[1]+2.5) for z in (a[2],b[2]+3.)])
            d=gpu.norm(camera);right=gpu.norm(np.cross([0,0,1],d));up=np.cross(d,right);ppm=960*.83/max(np.ptp(box@up),np.ptp(box@right)*960/1280)
            nv,lv=vn[no==bid],vl[lo==bid]
            old=next(r['grammar'] for r in old_report['cases'] if r['id']==key)
            assert len(nv)//3==old['near'] and len(lv)//3==old['lite'] and len(nv)//3<=1800
            A,ma=draw(nv,center,ppm,camera,'day',False);B,mb=draw(nv,center,ppm,camera,'day',True);C,mc=draw(lv,center,ppm,camera,'day',True)
            np.testing.assert_array_equal(ma,mb)
            diff=np.any(np.asarray(A)!=np.asarray(B),axis=2);assert not np.any(diff&~ma)
            assigned=stats['buildings'].get(bid,0)
            for tag,im in zip(('32','33','33_lite'),(A,B,C)):im.save(out/f'{key}_{tag}.png')
            iou=float((mb&mc).sum()/max(1,(mb|mc).sum()));assert iou>=.94
            rows.append(dict(id=key,near=len(nv)//3,lite=len(lv)//3,assigned_windows=assigned,changed_pixels=int(diff.sum()),near_lite_iou=iou,near_mask_identical=True))
            image_rows.append((key,A,B,C))
            if key in ('walkup_sections_0','section_slab_0','tower_stacks_0','tall_bay_tower_1'):
                rec=next(r for r in records if r['id']==key)
                w=max((w for w in rec['architecture']['facades'] if not w['party'] and not w['courtyard']),key=lambda w:w['length'])
                T=(np.asarray(w['end'])-w['start'])/w['length'];N=np.asarray(w['outward'])
                target=np.r_[np.asarray(w['start'])+T*w['length']*.38,min(spec['eave']*.55,9.)]+off
                view=(*list(N+.28*T),.14)
                sheet=Image.new('RGB',(2560,1990),(236,238,240));dd=ImageDraw.Draw(sheet)
                for k,mode in enumerate(('day','night')):
                    x,m=draw(nv,target,90.,view,mode,False);y,n=draw(nv,target,90.,view,mode,True)
                    np.testing.assert_array_equal(m,n)
                    if mode=='night':np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
                    for j,im in enumerate((x,y)):
                        sheet.paste(im,(j*1280,k*995));dd.text((j*1280+8,k*995+965),f'{key} | {mode} | Stage {32+j}',fill='black')
                sheet.save(out/f'{key}_closeup.png');closeups.append(key)
        del renderer
    for part in range(4):
        subset=image_rows[part*4:(part+1)*4]
        sheet=Image.new('RGB',(1260,len(subset)*286),(236,238,240));d=ImageDraw.Draw(sheet)
        for i,(key,*ims) in enumerate(subset):
            for j,im in enumerate(ims):
                im=im.copy();im.thumbnail((416,256));sheet.paste(im,(j*420+(420-im.width)//2,i*286))
                d.text((j*420+5,i*286+258),key+' | '+('32','33','33 Lite')[j],fill='black')
        sheet.save(out/f'comparison_{part}.png')
    assert len(rows)==16 and sum(r['near'] for r in rows)==old_report['near_total'] and sum(r['lite'] for r in rows)==old_report['lite_total']
    assert sum(a['windows'] for a in assignments_report)>0 and sum(r['changed_pixels'] for r in rows)>0
    report=dict(schema='pcg.research.stage33-overlay.v1',source_verified_files=source_files,atlas_verified_files=atlas_files,donor_layers=len(cat),base_layers=base,
        assignments=assignments_report,cases=rows,closeups=closeups,extra_triangles=0,near_total=sum(r['near'] for r in rows),lite_total=sum(r['lite'] for r in rows),
        limits=['Daytime donor-informed procedural frames/glass and bounded wall detail; paired balconies, entry and roof materials unchanged.','Roughness-derived crops are hypotheses; no certified semantic segmentation.','No native ABI change; fourth/fifth GPU opening texels are a research sidecar.','Same Stage32 meshes, not new GIS building parts. No native UNIGINE runtime or performance evidence.'])
    dump(out/'report.json',report);dump(out/'bundle.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='bundle.json'})
    print('INTEGRATED33',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','atlas','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.source,a.atlas,a.out)
