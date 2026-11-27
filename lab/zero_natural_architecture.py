"""Stage 28: actual stage-27 meshes + bounded detail modules and appearance.
Reuses the existing GLB exporter, PCGF packer, software EGL and exact cameras.
"""
from collections import Counter
from pathlib import Path
import hashlib,json,math,shutil,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
import detail28
import compile_zero_geometry_tile as tile
from compile_zero_tiles import pack_table,LAYERS,new_float32_audit
import zero_combined_lod as combined
from zero_window_entrance_stage import retain
import zero_photoreal as previous
sys.path.insert(0,str(Path(__file__).resolve().parent/'material_lab'))
import appearance28 as ap
import unified_metadata
import scene_io
import gpu_lab as gpu

SOURCE=Path('artifacts/zero-photoreal-architecture')
PREV26=Path('artifacts/zero-unified-architecture')
OUT=Path('artifacts/zero-natural-architecture')
COPY=('plans.json','facade_cells.bin','opening_blocks_v2.json','appearance_profiles.json','roof_balcony_cutout.glb','antenna_cutout.glb')


def dump(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def groups_from(path):
    groups={}
    for p,a,f,_ in scene_io.glb(path):
        role=p['extras']['pcg_role'];uv=a['TEXCOORD_0'].copy();uv[:,1]=1-uv[:,1]
        groups[role]=dict(v=scene_io.z_up(a['POSITION']).astype(np.float32),n=scene_io.z_up(a['NORMAL']).astype(np.float32),uv0=uv,uv1=a['TEXCOORD_1'].copy(),color=a['COLOR_0'].copy(),f=f.reshape(-1,3).astype(np.uint32))
    assert set(groups)==set(tile.GROUPS);return groups


def group_ids(groups):
    return {int(i) for g in groups.values() for i in g['color'][:,0].astype(int)+256*g['color'][:,1].astype(int)}


def assert_prefix(new,old):
    for role,g in old.items():
        for k,a in g.items():
            b=new[role][k][:len(a)]
            np.testing.assert_array_equal(a,b)


def build_geometry(records,placements,table,oldscene,ops,meta,profiles,prior):
    modules,stats,checks=detail28.plan(records,placements,oldscene,ops,meta,profiles,prior)
    again,stats2,checks2=detail28.plan(list(reversed(records)),placements,oldscene,ops,meta,profiles,prior)
    assert stats==stats2 and checks==checks2 and len(modules)==len(again)
    for a,b in zip(modules,again):
        assert {k:v for k,v in a.items() if k!='geometry'}=={k:v for k,v in b.items() if k!='geometry'}
        for (k,m,t),(K,M,T) in zip(a['geometry'],b['geometry']):assert (k,m)==(K,M);np.testing.assert_array_equal(t,T)
    g=groups_from(SOURCE/'tile_opaque.glb');old={role:{k:v.copy() for k,v in a.items()} for role,a in g.items()};lite=groups_from(SOURCE/'tile_lite_opaque.glb');dead=group_ids(g)-group_ids(lite)
    by={r['footprint']['building_id']:r for r in records};t=list(table);newplacements=list(placements);newrows=[]
    for m in modules:
        start=len(t);rec=by[m['building']];c=dict(building_id=m['building'],offset=[0,0,0],eave_m=rec['decision']['eave_m'],floors=len(rec['architecture']['floor_levels'])-1)
        combined.append_compact(g,t,c,m)
        if not m['primary']:dead.update(range(start,len(t)))
        pts=np.concatenate([q.reshape(-1,3) for _,_,q in m['geometry']]);item={k:v for k,v in m.items() if k!='geometry'};item.update(triangles=sum(len(q) for _,_,q in m['geometry']),bounds_world_local=[pts.min(0).tolist(),pts.max(0).tolist()],record_range=[start,len(t)])
        newplacements.append(item);newrows.append(item)
    assert t[:len(table)]==table;assert_prefix(g,old)
    lite_groups,_=retain(g,dead);assert_prefix(lite_groups,lite)
    for name,q in (('tile_opaque.glb',g),('tile_lite_opaque.glb',lite_groups)):
        p=OUT/name;tile.export_glb(q,p);tile.read_glb(p,q,t)
    return t,newplacements,newrows,stats,checks


def draw_probe(renderer,fragment,w,h,uniforms=None):
    gl,C,I,U,P=gpu.gl,gpu.C,gpu.I,gpu.U,gpu.P
    vert='#version 330 core\nvoid main(){vec2 p=gl_VertexID==0?vec2(-1,-1):gl_VertexID==1?vec2(3,-1):vec2(-1,3);gl_Position=vec4(p,0,1);}'
    p=gl('glCreateProgram',U)()
    for typ,src in ((0x8B31,vert),(0x8B30,fragment)):gl('glAttachShader',None,U,U)(p,gpu.shader(typ,src))
    gl('glLinkProgram',None,U)(p);ok=I();gl('glGetProgramiv',None,U,U,P)(p,0x8B82,C.byref(ok));assert ok.value
    fb,_,_=renderer.framebuffer(w,h);gl('glUseProgram',None,U)(p);gl('glBindFramebuffer',None,U,U)(0x8D40,fb);gl('glViewport',None,I,I,I,I)(0,0,w,h)
    for name,i in (uniforms or {}).items():gl('glUniform1i',None,I,I)(gl('glGetUniformLocation',I,U,gpu.B)(p,name.encode()),i)
    gl('glDisable',None,U)(0x0B71);gl('glBindVertexArray',None,U)(renderer.vao);gl('glDrawArrays',None,U,I,I)(4,0,3)
    a=np.empty((h,w,4),np.float32);gl('glReadPixels',None,I,I,I,I,U,U,P)(0,0,w,h,0x1908,0x1406,gpu.ptr(a));assert gl('glGetError',U)()==0 and np.isfinite(a).all();assert np.all(a[:,:,3]==1.)
    gl('glUseProgram',None,U)(renderer.p);gl('glEnable',None,U)(0x0B71);gl('glDeleteProgram',None,U)(p)
    return a


def ray_probe(renderer):
    a=draw_probe(renderer,gpu.fragment_source(photoreal=True,natural=True,natural_probe=True),8,4);expected=np.zeros((4,8));lo=np.array([-.75,-.60,.1]);hi=np.array([.85,.90,2.2]);hits=0
    for j in range(4):
        for i in range(8):
            p=np.array([i-2.5,j-1.5,.5]);d=np.array([j-1.5,i-2.5,.35]);d/=np.linalg.norm(d)
            if i>=6:p=np.array([j-1.5,-2.,.5]);d=np.array([0.,1. if i==6 else -1.,0.])
            # Independent per-plane intersections: choose nearest positive
            # face hit that lies inside all other slab intervals.
            q=[]
            for k in range(3):
                if abs(d[k])<1e-10:continue
                for b in (lo[k],hi[k]):
                    t=(b-p[k])/d[k];x=p+t*d
                    if t>.001 and np.all(x>=lo-1e-10) and np.all(x<=hi+1e-10):q.append(t)
            expected[j,i]=min(q) if q else -1.;hits+=bool(q)
    assert 0<hits<32;np.testing.assert_allclose(a[:,:,0],expected,atol=.003,rtol=0)
    return dict(samples=32,hits=hits,max_error_m=float(np.max(abs(a[:,:,0]-expected))))


def quant_probe(renderer,arrays):
    frag='''#version 330 core
uniform sampler2DArray albedoMaps,normalMaps,ormMaps;out vec4 color;
void main(){int x=int(gl_FragCoord.x),j=int(gl_FragCoord.y),k=x%16;ivec3 q=ivec3((k*73+19)%512,(k*127+13)%512,j);
vec3 c=x<16?texelFetch(albedoMaps,q,0).rgb:x<32?texelFetch(normalMaps,q,0).rgb:texelFetch(ormMaps,q,0).rgb;color=vec4(c,1.);}
'''
    a=draw_probe(renderer,frag,48,len(arrays[0]),{'albedoMaps':2,'normalMaps':3,'ormMaps':4});errors=[]
    for j in range(3):
        expect=np.stack([arrays[j][:,(k*127+13)%512,(k*73+19)%512,:3] for k in range(16)],axis=1);err=float(np.max(abs(a[:,16*j:16*(j+1),:3]-expect)));assert err<(.006 if j==0 else .0026),(j,err);errors.append(err)
    return dict(samples=int(len(arrays[0])*48),max_abs_error=errors,actual_formats=renderer.material_storage)


def render(near,lite,no,lo,oldscene,oldowner,meta,ops,proxy,records,newrows,prior):
    arrays,mips,info,coverage=previous.materials(SOURCE,OUT);spectral=ap.replace_maps(arrays,OUT)
    limit=gpu.I();gpu.gl('glGetIntegerv',None,gpu.U,gpu.P)(0x0D33,gpu.C.byref(limit));assert len(meta)<=limit.value and len(ops)<=limit.value
    vn=gpu.vertex_data(near);vl=gpu.vertex_data(lite);vo=gpu.vertex_data(oldscene);renderer=gpu.GL(vn,meta,ops,arrays,photoreal=True,cutout_mips=mips,natural=True,compact_materials=True)
    renderer.tex2(proxy,9,'reflectionProxies',False);optical=previous.optical_probe(renderer);ray=ray_probe(renderer);quant=quant_probe(renderer,arrays);del arrays,mips
    info['compact_material_storage']=renderer.material_storage;rows=[];masks={};by_id={r['id']:r['footprint']['building_id'] for r in records};by_bid={r['footprint']['building_id']:r for r in records}
    def draw(name,verts,center,ppm,view,mode,res,baseline=None):
        renderer.replace_vertices(verts);p=verts[:,:3];renderer.shadow_center=(p.min(0)+p.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(p,axis=0))*1.2))
        im,metrics=renderer.render(center,ppm,view,mode,tuple(res),info);assert im.size==tuple(res);im.save(OUT/(name+'.png'));mask=renderer.last_mask.copy()
        row=dict(name=name,mode=mode,resolution=list(res),center=list(center),ppm=ppm,view=list(view),**metrics)
        if baseline is not None:
            old=np.asarray(Image.open(SOURCE/(baseline['name']+'.png')).convert('RGB'));new=np.asarray(im);assert old.shape==new.shape
            row['changed_pixels']=int(np.any(old!=new,axis=2).sum());row['prior_foreground_pixels']=baseline['foreground_pixels'];assert row['changed_pixels']>0
        rows.append(row);return mask
    for row in prior['views']:
        name=row['name'];verts=vn
        if name=='neighborhood_lite':verts=vl
        elif name!='neighborhood_near':
            key='glass_grid_a' if name.startswith('office_upper_') else next(k for k in by_id if name.startswith(k+'_'));verts=vn[no==by_id[key]]
        mask=draw(name,verts,row['center'],row['ppm'],row['view'],row['mode'],row['resolution'],row)
        if name.endswith('_detail') or name.endswith('_detail_night') or name.startswith(('neighborhood_','office_upper_')):masks[name]=mask
    for name in ('panel_slab_a','brick_walkup_a','glass_grid_a'):np.testing.assert_array_equal(masks[name+'_detail'],masks[name+'_detail_night'])
    np.testing.assert_array_equal(masks['office_upper_front'],masks['office_upper_night'])
    a,b=masks['neighborhood_near'],masks['neighborhood_lite'];iou=float(np.count_nonzero(a&b)/np.count_nonzero(a|b));assert iou>=.95
    # Same stage-28 appearance on both actual meshes to isolate geometry.
    geometry=[];sheet=Image.new('RGB',(960,3*380),(235,237,239));d=ImageDraw.Draw(sheet)
    for i,(kind,preferred) in enumerate((('entry_cap','baltic_timber_a'),('balcony_edge','brick_walkup_a'),('sill','panel_slab_a'))):
        m=next((x for x in newrows if x['kind']==kind and x['building']==by_id[preferred]),None) or next(x for x in newrows if x['kind']==kind);bid=m['building'];w=next(x for x in by_bid[bid]['architecture']['facades'] if x['id']==m['facade']);t=(np.array(w['end'])-w['start'])/w['length'];n=np.asarray(w['outward']);view=[*list(n+.42*t),.30];center=np.asarray(m['bounds_world_local']).mean(0).tolist();ims=[]
        for j,(label,verts) in enumerate((('baseline',vo[oldowner==bid]),('candidate',vn[no==bid]))):
            name='geometry_'+kind+'_'+label;draw(name,verts,center,180.,view,'day',(1280,960));im=Image.open(OUT/(name+'.png')).convert('RGB');ims.append(np.asarray(im));im.thumbnail((480,360));sheet.paste(im,(j*480,i*380));d.text((j*480+5,i*380+362),kind+' | '+label,fill='black')
        changed=int(np.any(ims[0]!=ims[1],axis=2).sum());assert changed>0;geometry.append(dict(kind=kind,variant=m['variant'],building=bid,changed_pixels=changed))
    sheet.save(OUT/'geometry_comparison.png')
    return rows,info,optical,ray,quant,iou,spectral,geometry,coverage


def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((SOURCE/'bundle.json').read_text());prior=json.loads((SOURCE/'report.json').read_text());prior26=json.loads((PREV26/'report.json').read_text())
    for k,d in manifest.items():assert sha(SOURCE/k)==d
    for name in COPY:shutil.copyfile(SOURCE/name,OUT/name);assert sha(OUT/name)==manifest[name]
    records=json.loads((SOURCE/'plans.json').read_text());placements=json.loads((SOURCE/'placements.json').read_text());table=json.loads((SOURCE/'parameters.json').read_text());profiles=json.loads((SOURCE/'appearance_profiles.json').read_text());oldremap=json.loads((SOURCE/'cutout_records.json').read_text());old=np.load(SOURCE/'gpu_metadata.npz');oldmeta,ops=old['metadata'],old['openings']
    oldscene,oldowner,oldcuts,oldnv,ncheck=previous.load_scene(SOURCE,'tile_opaque.glb',table,previous.envelopes(records,placements));assert ncheck==prior['near_triangles']
    t,newplacements,newrows,stats,checks=build_geometry(records,placements,table,oldscene,ops,oldmeta,profiles,prior26)
    proxy,proxy_manifest,proxy_ids=ap.proxies(records,profiles);rev,rev_manifest,_=ap.proxies(list(reversed(records)),profiles);np.testing.assert_array_equal(proxy,rev);assert proxy_manifest==rev_manifest
    meta,remap=ap.extend_metadata(oldmeta,t,len(table),oldremap,profiles,proxy_ids,unified_metadata.LAYERS);delta=len(t)-len(table)
    oldscene,oldface=previous.remap_scene(oldscene,oldcuts,remap,oldnv);np.testing.assert_array_equal(oldface,old['cutout_face_records'].astype(np.int64)+delta)
    bounds=previous.envelopes(records,newplacements);near,no,cn,nv,ncheck=previous.load_scene(OUT,'tile_opaque.glb',t,bounds);lite,lo,cl,lv,lcheck=previous.load_scene(OUT,'tile_lite_opaque.glb',t,bounds)
    near,face=previous.remap_scene(near,cn,remap,nv);lite,facelite=previous.remap_scene(lite,cl,remap,lv);np.testing.assert_array_equal(face,oldface);np.testing.assert_array_equal(face,facelite)
    assert ncheck==sum(x['near'] for x in stats.values()) and lcheck==sum(x['lite'] for x in stats.values()) and ncheck-prior['near_triangles']==checks['added']
    for bid,x in stats.items():assert np.count_nonzero(no==bid)//3==x['near'] and np.count_nonzero(lo==bid)//3==x['lite']
    cells=np.frombuffer((OUT/'facade_cells.bin').read_bytes(),np.uint8);audit=new_float32_audit();samples=pack_table(t,cells,OUT/'facade_params.bin',audit);oldabi=(SOURCE/'facade_params.bin').read_bytes();assert (OUT/'facade_params.bin').read_bytes()[16:len(oldabi)]==oldabi[16:]
    cpp=json.loads(subprocess.run([decoder,str(OUT/'facade_params.bin'),str(OUT/'facade_cells.bin'),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout);assert cpp['records']==len(t) and cpp['cells']==len(cells)
    dump(OUT/'parameters.json',t);dump(OUT/'placements.json',newplacements);dump(OUT/'detail_placements.json',newrows);dump(OUT/'reflection_proxies.json',proxy_manifest);dump(OUT/'cutout_records.json',[dict(building=bid,kind=kind,id=j) for (bid,kind),j in sorted(remap.items())]);np.savez_compressed(OUT/'gpu_metadata.npz',metadata=meta,openings=ops,cutout_face_records=face,reflection_proxies=proxy)
    rows,info,optical,ray,quant,iou,spectral,geometry,coverage=render(near,lite,no,lo,oldscene,oldowner,meta,ops,proxy,records,newrows,prior);previous.sheets(SOURCE,OUT,records,labels=('27','28'))
    counts=Counter(x['kind'] for x in newrows);variants=Counter(x['variant'] for x in newrows);assert counts['entry_cap']==36 and counts['balcony_edge']>=240 and counts['sill']>=300 and len(variants)==8
    report=dict(schema='pcg.research.natural-architecture.v1',buildings=len(records),archetypes=len({p['archetype'] for p in profiles.values()}),near_triangles=ncheck,lite_triangles=lcheck,max_building_triangles=max(x['near'] for x in stats.values()),source_geometry_preserved=True,source_manifest_verified=len(manifest),canonical_cells_identical=True,opening_metadata_identical=True,cutout_geometry_identical=True,existing_abi_record_payload_identical=True,input_order_invariant=True,details=dict(counts),variants=dict(variants),placement_checks=checks,per_building=stats,cpp=cpp,abi_samples=samples,appearance_records=len(meta),optical_probe=optical,reflection_proxy_probe=ray,quantization_probe=quant,spectral=spectral,geometry_comparison=geometry,near_lite_iou=iou,day_night_mask_exact=True,materials=info,alpha_coverage=coverage,views=rows,limitations=['Actual EGL software renderer, no native UNIGINE or GPU benchmark.','Isotropic periodic shared microtexture plus decorrelated shader phases reduces repetition, not measured photographic textures.','Four nearest canonical bounding boxes per building are coarse reflection proxies; no exact geometry reflection, HDRI or GI.','Virtual room furniture and occupancy-controlled balcony bounce are analytic shading approximations.','Geometry adds canopy caps, balcony front-edge profiles and grouped sills, not new balcony volumes; budget rejects other whole groups.','Logical material texture storage halves, not measured VRAM/driver compression; added sample cost near remains.','Synthetic building plans, no real-GIS regional calibration; new shader and metadata adapter still research only.'])
    assert report['buildings']==36 and report['archetypes']==15 and report['max_building_triangles']<=1800 and len(rows)==len(prior['views'])+6
    dump(OUT/'report.json',report);dump(OUT/'bundle.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='bundle.json'});print('STAGE28',json.dumps({k:v for k,v in report.items() if k not in ('views','per_building','alpha_coverage','limitations')}),flush=True)


if __name__=='__main__':main(sys.argv[1])
