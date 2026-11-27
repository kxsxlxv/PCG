"""Stage 29: private replay of verified stage-28 geometry with imported PBR.
The normal CI lane uses synthetic maps only. No private textures in GitHub.
Run from the repo root, after extracting the stage-28 CI artifact to artifacts/.
"""
import argparse, hashlib, json, os, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent/'material_lab'))
import external29 as ext
import zero_photoreal as prev
import zero_natural_architecture as stage28
import appearance28 as ap
import gpu_lab as gpu

def fixture_data(size=512):
    # Original, synthetic deterministic texture fixture; no donor assets.
    y,x=np.mgrid[:size,:size]/size
    a=np.stack((.15+.45*x,.17+.5*y,.24+.12*np.sin(x*16)),axis=-1).astype(np.float32)
    n=np.broadcast_to([.5,.5,1.],a.shape).astype(np.float32).copy()
    r=np.broadcast_to([.8,.65,0],a.shape).astype(np.float32).copy()
    return {k:(a.copy(),n.copy(),r.copy()) for k in ext.SOURCES},{k:{'sha256':'synthetic-fixture'} for k in ext.SOURCES}

def fixture_import(out):
    # Distinct known channels, rectangular source, no AO/metal maps.
    p=out/'fixture.zip'
    a=np.zeros((4,8,3),np.uint8);a[:]=[128,64,32]
    n=np.zeros_like(a);n[:]=[128,128,255];r=np.full((4,8),192,np.uint8)
    with zipfile.ZipFile(p,'w') as z:
        for name,value in [('x_Color.png',a),('x_NormalGL.png',n),('x_Roughness.png',r)]:
            import io
            b=io.BytesIO();Image.fromarray(value).save(b,format='PNG');z.writestr(name,b.getvalue())
    maps,info=ext.load_archive(p,8)
    np.testing.assert_allclose(maps[0][0,0],ext.linear(np.array([128,64,32])/255),atol=1e-6)
    np.testing.assert_allclose(maps[2][0,0],[1,192/255,0],atol=1e-6)
    assert info['source_size']==[8,4] and info['fallback_ao'] and info['fallback_metal']
    p.unlink()

def run(source,out,private=None):
    if out.resolve()==source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError('output must be separate from immutable baseline')
    if private is not None and os.getenv('GITHUB_ACTIONS')=='true':
        raise RuntimeError('private data prohibited in CI')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'bundle.json').read_text())
    for name,digest in manifest.items():
        if ext.sha(source/name)!=digest:raise ValueError(('baseline digest',name))
    prior=json.loads((source/'report.json').read_text())
    records=json.loads((source/'plans.json').read_text())
    table=json.loads((source/'parameters.json').read_text())
    placements=json.loads((source/'placements.json').read_text())
    remap={(r['building'],r['kind']):r['id'] for r in json.loads((source/'cutout_records.json').read_text())}
    old=np.load(source/'gpu_metadata.npz');metadata=old['metadata'];ops=old['openings'];proxy=old['reflection_proxies']
    bounds=prev.envelopes(records,placements)
    near,no,cn,nv,ncheck=prev.load_scene(source,'tile_opaque.glb',table,bounds)
    lite,lo,cl,lv,lcheck=prev.load_scene(source,'tile_lite_opaque.glb',table,bounds)
    near,face=prev.remap_scene(near,cn,remap,nv);lite,face_lite=prev.remap_scene(lite,cl,remap,lv)
    np.testing.assert_array_equal(face,old['cutout_face_records']);np.testing.assert_array_equal(face_lite,face)
    assert ncheck==prior['near_triangles'] and lcheck==prior['lite_triangles']
    baseline_dir=out/'baseline_bank';baseline_dir.mkdir(exist_ok=True)
    arrays,mips,info,coverage=prev.materials(source,baseline_dir);ap.replace_maps(arrays,baseline_dir)
    vn=gpu.vertex_data(near);vl=gpu.vertex_data(lite)
    renderer=gpu.GL(vn,metadata,ops,arrays,photoreal=True,cutout_mips=mips,natural=True,compact_materials=True)
    renderer.tex2(proxy,9,'reflectionProxies',False)
    by_id={r['id']:r['footprint']['building_id'] for r in records}
    names=['panel_slab_a_detail','brick_walkup_a_detail','baltic_timber_a_roof','central_townhouse_a_iso_gpu','glass_grid_a_detail','neighborhood_near','neighborhood_lite']
    views=[r for r in prior['views'] if r['name'] in names]
    if len(views)!=len(names):raise ValueError(('missing views',set(names)-{r['name'] for r in views}))
    def draw(row,variant):
        name=row['name'];verts=vl if name=='neighborhood_lite' else vn if name=='neighborhood_near' else vn[no==by_id[next(k for k in by_id if name.startswith(k+'_'))]]
        renderer.replace_vertices(verts);p=verts[:,:3]
        renderer.shadow_center=(p.min(0)+p.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(p,axis=0))*1.2))
        im,metrics=renderer.render(row['center'],row['ppm'],row['view'],row['mode'],tuple(row['resolution']),info)
        im.save(out/(name+'_'+variant+'.png'))
        return np.asarray(im),renderer.last_mask.copy(),metrics
    baseline={row['name']:draw(row,'baseline') for row in views}
    if private is None:
        fixture_import(out);data,source_manifest=fixture_data()
    else:data,source_manifest=ext.load_all(private,out)
    candidate,rows=ext.apply(arrays,metadata,data,source_manifest,out)
    # Shared IDs, geometry, cutout and room resources remain unchanged.
    renderer.tex2(candidate,0,'metadata',False)
    for j,k in enumerate(('albedoMaps','normalMaps','ormMaps')):renderer.texarray(arrays[j],j+2,k)
    renderer.uniform('externalPbrLayers',ext.AUTHORED_MASK)
    quant=stage28.quant_probe(renderer,arrays)
    # Quant probe restores the original program; explicit uniform assignment.
    renderer.uniform('externalPbrLayers',ext.AUTHORED_MASK)
    results=[];sheet=Image.new('RGB',(960,len(views)*305),(232,234,236));d=ImageDraw.Draw(sheet)
    for i,row in enumerate(views):
        name=row['name'];a,mask,metrics=draw(row,'candidate');b,bmask,_=baseline[name]
        np.testing.assert_array_equal(mask,bmask)
        count=int(np.count_nonzero(np.any(a!=b,axis=2)&mask))
        for j,img in enumerate((b,a)):
            im=Image.fromarray(img);im.thumbnail((470,276));sheet.paste(im,(j*480+(480-im.width)//2,i*305))
            d.text((j*480+8,i*305+280),name+' | '+('stage 28' if j==0 else 'private PBR' if private else 'synthetic fixture'),fill='black')
        results.append(dict(name=name,foreground_pixels=int(mask.sum()),changed_pixels=count,**{'render_metrics':metrics}))
    assert any(r['changed_pixels']>0 for r in results)
    sheet.save(out/'comparison.png')
    np.savez_compressed(out/'gpu_metadata_overlay.npz',metadata=candidate,openings=ops,cutout_face_records=face,reflection_proxies=proxy)
    report=dict(schema='pcg.research.reference-pbr.v1',source_kind='private' if private else 'synthetic',
        verified_baseline_files=len(manifest),near_triangles=ncheck,lite_triangles=lcheck,geometry_unchanged=True,
        materials=len(data),mapped_layers=len(rows),mapped_sources=len({r['source'] for r in rows}),
        cutouts_and_openings_unchanged=True,silhouettes_exact=True,quantization_probe=quant,views=results,
        limits=['Private input replay is not a native UNIGINE material adapter.','Repeat widths are authored estimates.','Paving, peeling painted metal and structured concrete tile facade are normalized but not mapped to existing layers.','No private geometry or textures are redistributed.'])
    ext.dump(out/'report.json',report);print('STAGE29',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('artifacts/zero-natural-architecture'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--private-materials',type=Path)
    args=p.parse_args()
    run(args.source,args.out,args.private_materials)
