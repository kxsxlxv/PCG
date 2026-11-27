"""Stage 29 final research replay: original equipment + exact PBR overlay.
Private overlays are consumed only offline; the public CI uses synthetic inputs.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import equipment29 as eq
import zero_reference_assets as pbr
from zero_reference_assets import ext,prev,ap,gpu
from quant_probe import quant_probe

COPY=('roof_balcony_cutout.glb','antenna_cutout.glb','facade_params.bin','facade_cells.bin','opening_blocks_v2.json','plans.json','parameters.json','cutout_records.json')

def run(source,overlay,out):
    out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    prior=read(source/'report.json');preport=read(overlay/'report.json')
    if os.getenv('GITHUB_ACTIONS')=='true' and preport['source_kind']!='synthetic':raise RuntimeError('private overlay prohibited in CI')
    manifest=read(source/'bundle.json')
    for name,digest in manifest.items():assert ext.sha(source/name)==digest,name
    records=read(source/'plans.json');placements=read(source/'placements.json');table=read(source/'parameters.json')
    remap={(r['building'],r['kind']):r['id'] for r in read(source/'cutout_records.json')}
    old=np.load(source/'gpu_metadata.npz');data=np.load(overlay/'gpu_metadata_overlay.npz');meta=data['metadata'];ops=data['openings'];proxy=data['reflection_proxies']
    for k in ('openings','cutout_face_records','reflection_proxies'):np.testing.assert_array_equal(old[k],data[k])
    keep=np.ones(meta.shape,bool);keep[:,0,2]=False;np.testing.assert_array_equal(meta[keep],old['metadata'][keep])
    geom=out/'geometry';result=eq.build(source,geom,records,placements,table,prior)
    # Reverse all input iteration order and verify identical exported mesh bytes.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rev=eq.build(source,Path(tmp),list(reversed(records)),list(reversed(placements)),table,prior)
        assert rev==result
        for name in ('tile_opaque.glb','tile_lite_opaque.glb'):assert ext.sha(geom/name)==ext.sha(Path(tmp)/name),name
    for k in COPY:assert ext.sha(source/k)==ext.sha(geom/k)
    bounds=prev.envelopes(records,placements)
    def load(root,name):
        scene,owner,cuts,nv,count=prev.load_scene(root,name,table,bounds);scene,face=prev.remap_scene(scene,cuts,remap,nv);np.testing.assert_array_equal(face,old['cutout_face_records']);return scene,owner,count
    sn,on,cn=load(source,'tile_opaque.glb');nn,no,nc=load(geom,'tile_opaque.glb');ll,lo,lc=load(geom,'tile_lite_opaque.glb')
    assert cn==prior['near_triangles'] and nc==result['near_triangles'] and lc==result['lite_triangles']
    for bid,x in result['per_building'].items():assert np.count_nonzero(no==bid)//3==x['near'] and np.count_nonzero(lo==bid)//3==x['lite']
    bank=out/'baseline_bank';bank.mkdir(exist_ok=True)
    arrays,mips,info,_=prev.materials(source,bank);ap.replace_maps(arrays,bank)
    hashes={}
    for i in ext.SLOTS:
        path=overlay/'material_bank'/str(i)
        for k,name in enumerate(('basecolor','normal','orm')):
            p=path/(name+'.png');hashes[str(p.relative_to(overlay))]=ext.sha(p);a=np.asarray(Image.open(p).convert('RGB'),np.float32)/255.
            if k==0:arrays[k][i,:,:,:3]=ext.linear(a);arrays[k][i,:,:,3]=1
            else:arrays[k][i]=a
    vs,vn,vl=map(gpu.vertex_data,(sn,nn,ll));renderer=gpu.GL(vn,meta,ops,arrays,photoreal=True,cutout_mips=mips,natural=True,compact_materials=True)
    renderer.tex2(proxy,9,'reflectionProxies',False);renderer.uniform('externalPbrLayers',ext.AUTHORED_MASK);quant=quant_probe(renderer,arrays)
    def draw(verts,center,ppm,view,mode,res):
        renderer.replace_vertices(verts);p=verts[:,:3];renderer.shadow_center=(p.min(0)+p.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(p,axis=0))*1.2))
        im,metrics=renderer.render(center,ppm,view,mode,tuple(res),info);return im,renderer.last_mask.copy(),metrics
    views=[];examples=[]
    for variant in sorted(result['variants']):
        row=next(r for r in result['placements'] if r['accepted'] and r['variant']==variant);bid=row['building'];examples.append((variant,row['center'],500. if row['kind']=='ac' else 340.,row['view'],'day',(960,640),bid))
    r=next(r for r in prior['views'] if r['name']=='neighborhood_near');examples.append(('neighborhood',r['center'],r['ppm'],r['view'],r['mode'],r['resolution'],None))
    sheet=Image.new('RGB',(1260,len(examples)*302),(232,234,236));d=ImageDraw.Draw(sheet)
    for i,(name,center,ppm,view,mode,res,bid) in enumerate(examples):
        pics=[];masks=[];metrics=[]
        for j,(label,v,owner) in enumerate((('prior_near',vs,on),('new_near',vn,no),('new_lite',vl,lo))):
            im,mask,m=draw(v if bid is None else v[owner==bid],center,ppm,view,mode,res);im.save(out/(name+'_'+label+'.png'));pics.append(np.asarray(im));masks.append(mask);metrics.append(m)
            im.thumbnail((416,275));sheet.paste(im,(j*420+(420-im.width)//2,i*302));d.text((j*420+5,i*302+277),name+' | '+label,fill='black')
        changed=int(np.any(pics[0]!=pics[1],axis=2).sum());assert changed>0,name
        iou=float(np.count_nonzero(masks[1]&masks[2])/np.count_nonzero(masks[1]|masks[2]))
        if name=='neighborhood':assert iou>=.95
        views.append(dict(name=name,building=bid,changed_pixels=changed,near_lite_iou=iou,metrics=metrics))
    sheet.save(out/'equipment_comparison.png');result.update(source_kind=preport['source_kind'],baseline_files_verified=len(manifest),overlay_map_hashes=hashes,geometry_deterministic=True,unchanged_inputs=list(COPY),quantization_probe=quant,views=views)
    ext.dump(out/'report.json',result)
    ext.dump(out/'bundle.json',{str(p.relative_to(out)):ext.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='bundle.json'})
    print('STAGE29_EQUIPMENT',json.dumps({k:v for k,v in result.items() if k not in ('placements','per_building','overlay_map_hashes')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('artifacts/zero-natural-architecture'));p.add_argument('--overlay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.overlay,a.out)
