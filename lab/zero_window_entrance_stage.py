"""Single consolidated window/balcony/entrance stage for the visual lab.
New semantic sidecar is deliberately versioned separately from legacy PCGF.
Open CI never uploads commercial textures or derived renders.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,os,shutil,subprocess
import numpy as np
from PIL import Image,ImageDraw
from pcg.far_emission import srgb_encode
from compile_zero_tiles import pack_table,LAYERS
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_combined_lod as combined
import zero_integrated_cutouts as integrated
import zero_alpha_cutout as cut
import zero_facade_trim as trim
import zero_architecture_correctness as corr
import zero_window_blocks as blocks
from zero_window_materials import Library,WindowShader,pom,hit_box,perspective_uv

OUT=Path('artifacts/zero-window-entrance-stage')
PREV=Path('artifacts/zero-roof-refinement')
ROOT=Path(__file__).resolve().parents[1]/'assets/third_party'

def retain(groups,dead):
    g={};removed=0
    for role,q in groups.items():
        rid=ref.decode(q['color'][:,:2].astype(np.float32)/255);f=q['f'];keep=~np.isin(rid[f[:,0]],list(dead));removed+=int((~keep).sum());f=f[keep];used=np.unique(f);remap=np.full(len(rid),-1,np.int64);remap[used]=np.arange(len(used))
        g[role]={k:(remap[f].astype(v.dtype) if k=='f' else v[used].copy()) for k,v in q.items()}
        np.testing.assert_array_equal(q['v'][q['f'][keep]],g[role]['v'][g[role]['f']])
        np.testing.assert_array_equal(q['uv1'][q['f'][keep]],g[role]['uv1'][g[role]['f']])
    return g,removed

def author_maps(lib,maps,ppm):
    # Use source linear mip pyramid; existing material path expects sRGB.
    result=maps.copy()
    for key,m,repeat in [('red_brick',lib.brick[1],1.2),('concrete_panel',lib.concrete,2.4)]:
        level=int(np.clip(np.log2(max(m.levels[0].shape[:2])/max(1.,ppm*repeat)),0,len(m.levels)-1));result[key]=srgb_encode(m.levels[level])
    return result

def test_math():
    uv=np.array([[.25,.25],[.5,.5],[.75,.75],[.96,.5]])
    q,t,f=hit_box(uv,[0,0],2.4,1.2,1.5);np.testing.assert_allclose(q,uv);np.testing.assert_allclose(t,2.4);assert np.all(f==4)
    q,t,f=hit_box(uv,[.9,0],2.4,1.2,1.5);assert np.any(f==3) and np.all(t>=0)
    q,t,f=hit_box(uv,[0,-.8],2.4,1.2,1.5);assert np.any(f==1)
    p,k,d=pom(uv,[0,0],1.4,1.5,steps=20);np.testing.assert_allclose(p,uv);assert d[1]<d[0] and k[1]==2 and k[0]==3
    q,K,D=pom(uv,[.9,.13],1.4,1.5,steps=20);assert np.any(abs(q-p)>1e-3) and np.isfinite(D).all()
    a=perspective_uv(uv,[0,0],2.4,1.4,1.5);b=perspective_uv(uv,[.9,.13],2.4,1.4,1.5);assert np.any(abs(a-b)>1e-3)

def run(groups,table,cells,scene,actual,antenna,maps,previous,info,frames,visible,two,decoder):
    OUT.mkdir(parents=True,exist_ok=True);test_math()
    private=os.getenv('PCG_PRIVATE_INTERIORS')
    if private and os.getenv('GITHUB_ACTIONS'):raise RuntimeError('Commercial asset evaluation must run privately, outside GitHub Actions')
    lib=Library(ROOT,private)
    modules,guards,pairs,dead,entrances,checks=blocks.plan(scene,cells)
    m2,g2,p2,d2,e2,_=blocks.plan(scene,cells);assert p2==pairs and g2==guards and d2==dead and e2==entrances
    for m,n in zip(modules,m2):
        assert m['id']==n['id']
        for (_,_,a),(_,_,b) in zip(m['geometry'],n['geometry']):np.testing.assert_array_equal(a,b)
    # Test the original actual panel seam geometry against the widened paired
    # openings. No old seam can cross the new door/window or intervening post.
    seam_checks=0
    for r in table:
        if r.get('role')!='wall' or not r['building'].endswith('panel_block'):continue
        seams=corr.panel_joints(r,cells)
        for pair in pairs.get(str(r['id']),{}).values():
            for k in ('door','window','post'):
                for joint in seams:
                    assert not blocks.rectover(pair[k],joint),(r['facade_id'],k,pair[k],joint)
                    seam_checks+=1
    g,removed=retain(groups,dead);t=list(table)
    for m in modules:
        r=next(r for r in table if r.get('facade_id')==m['facade']);c=dict(building_id=m['building'],offset=[0,0,0],eave_m=r['height'],floors=r['floors'])
        combined.append_compact(g,t,c,m)
    path=OUT/'tile_opaque.glb';tile.export_glb(g,path);v,u,ids,_=tile.read_glb(path,g,t)
    assert t[:len(table)]==table
    samples=pack_table(t,cells,OUT/'facade_params.bin');(OUT/'facade_cells.bin').write_bytes(cells.tobytes());assert (OUT/'facade_cells.bin').read_bytes()==(PREV/'facade_cells.bin').read_bytes()
    check=json.loads(subprocess.run([decoder,str(OUT/'facade_params.bin'),str(OUT/'facade_cells.bin'),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert check['records']==len(t) and check['cells']==len(cells)
    strips=[cut.strip(q['path'],q['height'],cut.PERIOD['balcony']) for q in guards]
    # Export changed balcony rails, unchanged roof guard and unchanged antennas.
    meshes={'roof':(actual['roof'][0],np.c_[actual['roof'][1][:,0],1-actual['roof'][1][:,1]],actual['roof'][2]),'balcony':cut.join(strips)}
    masks={k:actual[k][3] for k in meshes};assert len(meshes['balcony'][2])==186
    p=OUT/'roof_balcony_cutout.glb';cut.export(p,meshes,masks);newactual=cut.read(p,meshes,masks)
    np.testing.assert_allclose(newactual['roof'][0],actual['roof'][0],atol=0);np.testing.assert_allclose(newactual['roof'][1],actual['roof'][1],atol=1e-7)
    candidate=(v,u,ids,t)
    for k,(vv,uu,ff,im) in newactual.items():candidate=integrated.append_layer(candidate,vv,uu,ff,k)
    vv,uu,ff,im=antenna;candidate=integrated.append_layer(candidate,vv,uu,ff,'antenna')
    shutil.copyfile(PREV/'antenna_cutout.glb',OUT/'antenna_cutout.glb')
    # New table contains unused legacy balcony records to keep stable semantic
    # indices; pair geometry and shader requirements are explicit in sidecar.
    trim.dump(OUT/'parameters.json',t);trim.dump(OUT/'opening_blocks_v2.json',dict(schema='pcg.window-blocks.v2-research',legacy_cells_unchanged=True,paired_openings=pairs,entrances=entrances))
    meta=[dict({k:v for k,v in m.items() if k!='geometry'},triangles=sum(len(p) for _,_,p in m['geometry'])) for m in modules];trim.dump(OUT/'placements.json',meta)
    total=len(candidate[0])//3;added=sum(len(p) for m in modules for _,_,p in m['geometry']);assert total==len(scene[0])//3-removed+added
    shader=WindowShader(previous,lib,cells,pairs,candidate,info,frames)
    # Guard protection, deterministic material math and real sample coverage.
    checked=0;shader.configure((.3,-1,.18),64,1280);am=author_maps(lib,maps,128)
    for r in candidate[3]:
        if r.get('role')=='wall' and 'accessory' not in r:
            s=np.random.default_rng(r['seed']).random((1024,2))*[r['length'],r['height']]
        elif r.get('accessory')=='stage21_glazed' and r['id'] in shader.axes:
            s=np.random.default_rng(r['seed']).random((128,2))*r['span']
        elif r.get('accessory','').startswith('cutout_'):s=np.array([[.15,.25],[.5,.75],[1.,1.]])
        else:continue
        for mode in ('day','night'):
            a=shader.shade(s,r,am,mode);b=shader.shade(s,r,am,mode)
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y);assert np.isfinite(x).all() and (x>=0).all()
            if r.get('accessory','').startswith('cutout_'):
                for x,y in zip(a,previous(s,r,am,mode)):np.testing.assert_array_equal(x,y)
        checked+=len(s)
    shader.counts.clear()
    cams=[('panel_full',(58,6,10.6),16.,(.35,-1,.22),'day',(1920,1080)),
          ('brick_full',(107,6,8.5),17.,(.32,-1,.22),'day',(1920,1080)),
          ('panel_detail',(50.4,0,10.5),88.,(.32,-1,.13),'day',(1280,1280)),
          ('panel_oblique',(50.4,0,10.5),88.,(.85,-1,.13),'day',(1280,1280)),
          ('night',(58,6,10.6),16.,(.35,-1,.22),'night',(1920,1080)),
          ('entrance',(15,0,1.35),95.,(.28,-1,.10),'day',(1280,1280)),
          ('entrance_night',(15,0,1.35),95.,(.28,-1,.10),'night',(1280,1280)),
          ('aerial',(61,6,10),8.,(.3,-1,1.65),'day',(1920,1080))]
    sheet=Image.new('RGB',(1400,370*len(cams)),(226,228,228));draw=ImageDraw.Draw(sheet);rows=[];original=ref.shader
    try:
        for i,(name,center,ppm,view,mode,res) in enumerate(cams):
            shader.configure(view,ppm,res[1]);auth=author_maps(lib,maps,shader.ppm)
            l0=trim.shadow(scene,maps,center,ppm,visible,two) if mode=='day' else None
            l1=trim.shadow(candidate,auth,center,ppm,visible,two) if mode=='day' else None
            ims=[]
            for j,(label,sc,fn,mp,light) in enumerate((('baseline',scene,previous,maps,l0),('candidate',candidate,shader.shade,auth,l1))):
                ref.shader=fn;im,_=ref.render(*sc,mp,mode,center_override=center,ppm_override=ppm,view_override=view,visibility=visible,two_sided=two,lighting=light,resolution=res)
                assert im.size==res;im.save(OUT/f'{name}_{label}.png');ims.append(np.asarray(im))
                thumb=im.copy();thumb.thumbnail((700,340));sheet.paste(thumb,(j*700+(700-thumb.width)//2,i*370+22));draw.text((j*700+8,i*370+5),name+' | '+label,fill='black')
            rows.append(dict(view=name,changed_rgb_pixels=int(np.any(ims[0]!=ims[1],axis=2).sum()),resolution=list(res)))
    finally:ref.shader=original
    sheet.save(OUT/'comparison.png');assert all(r['changed_rgb_pixels']>0 for r in rows)
    assert all(any(k.startswith(q) for k in shader.counts) for q in ('Panelka/photo','Gaxil/','Three-Fenestra/'))
    report=dict(schema='pcg.research.windows-entrances.v2',baseline_triangles=len(scene[0])//3,triangles=total,removed_balcony_triangles=removed,added_triangles=added,
        balcony_count=len(guards)+sum(m['kind']=='balcony_glazed' for m in modules),glazed_modules=sum(m['kind']=='balcony_glazed' for m in modules),open_modules=len(guards),partial_modules=0,
        paired_blocks=sum(len(p) for p in pairs.values()),slab_widths=dict(Counter(str(m['width_m']) for m in modules if 'width_m' in m)),entrances=entrances,
        clearance_checks=checks,panel_seam_checks=seam_checks,deterministic_samples=checked,cpp_legacy_abi_samples=samples,canonical_cells_unchanged=True,
        material_pixel_counts=shader.counts,private_maps_loaded=len(lib.private),private_decode_errors=lib.errors,views=rows,limitations=[
        'Paired window/door sidecar needs native runtime lookup/ABI v2; legacy ABI audit alone does not validate it.',
        'CPU view-dependent POM, cross atlas interior rays and approximate normal/roughness response, not production GPU PBR. Orthographic camera only.',
        'Enclosure glass is opaque interior-map approximation, not true transmission to the wall behind. Near LOD cannot reproduce real sill silhouette.',
        'Open CI uses only redistributable assets. Licensed EXR room paths are private local evaluation; fallback corridor for lobby.',
        'Three rectangular synthetic buildings; only the 3 actual ground door anchors, no arbitrary entrance/stairwell typology.',
        'Manual crop calibrations and material scale estimates, no native UNIGINE GPU benchmark; visual acceptance pending.'])
    trim.dump(OUT/'report.json',report)
    files=[path,p,OUT/'antenna_cutout.glb',OUT/'parameters.json',OUT/'facade_params.bin',OUT/'facade_cells.bin',OUT/'opening_blocks_v2.json',OUT/'placements.json'];trim.dump(OUT/'bundle.json',dict(schema=report['schema'],hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    print('WINDOW_ENTRANCE_REPORT '+json.dumps(report),flush=True)
