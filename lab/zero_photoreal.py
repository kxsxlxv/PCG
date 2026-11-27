"""Stage 27: rerender actual stage-26 GLBs with a shared appearance bank.
No host/attachment geometry edits, no new per-opening triangles, no SDK.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json, shutil, sys, math
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent/'material_lab'))
import scene_io
import gpu_lab as gpu
import appearance27 as ap

SOURCE=Path('artifacts/zero-unified-architecture')
OUT=Path('artifacts/zero-photoreal-architecture')
FILES=('tile_opaque.glb','tile_lite_opaque.glb','roof_balcony_cutout.glb','antenna_cutout.glb',
       'parameters.json','facade_cells.bin','facade_params.bin','opening_blocks_v2.json','plans.json','placements.json')


def dump(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def envelopes(records,placements):
    out={}
    for r in records:
        f=r['footprint'];off=np.asarray(r['offset']);p=np.asarray(f['exterior']);h=r['decision']['top_m']
        out[f['building_id']]=np.array([np.r_[p.min(0),0.]+off,np.r_[p.max(0),h]+off])
    for m in placements:
        box=out[m['building']];b=np.asarray(m['bounds_world_local']);box[0]=np.minimum(box[0],b[0]);box[1]=np.maximum(box[1],b[1])
    return out


def load_scene(root,opaque,table,bounds):
    v=[];u=[];ids=[];owners=[];cuts=[];checks=0
    for p,a,f,t in scene_io.glb(root/opaque):
        vv=scene_io.z_up(a['POSITION'])[f];uv=a['TEXCOORD_1'][f];c=a['COLOR_0'].astype(int);rid=(c[:,0]+256*c[:,1])[f]
        assert np.all(rid.reshape(-1,3)==rid.reshape(-1,3)[:,:1]) and rid.max()<len(table)
        own=np.array([table[j]['building'] for j in rid]);v.append(vv);u.append(uv);ids.append(rid);owners.append(own)
        for bid in set(own):
            b=bounds[bid];q=vv[own==bid];assert np.all(q>=b[0]-.0002) and np.all(q<=b[1]+.0002)
        checks+=len(vv)//3
    opaque_vertices=sum(len(x) for x in v)
    for path in ('roof_balcony_cutout.glb','antenna_cutout.glb'):
        for p,a,f,t in scene_io.glb(root/path):
            kind=t['materials'][p['material']]['name'].removeprefix('cutout_');assert kind in ('roof','balcony','antenna')
            vv=scene_io.z_up(a['POSITION'])[f];uv=a['TEXCOORD_0'][f];tri=vv.reshape(-1,3,3);own=np.full(len(tri),'',dtype='<U128');count=np.zeros(len(tri),int)
            lo,hi=tri.min(1),tri.max(1)
            for bid,b in bounds.items():
                inside=np.all(lo>=b[0]-.0002,axis=1)&np.all(hi<=b[1]+.0002,axis=1);count+=inside;own[inside]=bid
            assert np.all(count==1),(kind,np.unique(count,return_counts=True))
            rid=np.full(len(vv),len(table)+('roof','balcony','antenna').index(kind),int)
            start=sum(len(x) for x in v);cuts.append(dict(kind=kind,start=start,count=len(vv),owner=own.copy()))
            v.append(vv);u.append(uv);ids.append(rid);owners.append(np.repeat(own,3));checks+=len(tri)
    scene=(np.concatenate(v),np.concatenate(u),np.concatenate(ids),table);owner=np.concatenate(owners)
    assert len(scene[0])==len(owner) and len(scene[0])%3==0
    return scene,owner,cuts,opaque_vertices,checks


def remap_scene(scene,cuts,remap,opaque_vertices):
    v,u,ids,t=scene;new=ids.copy();pairs=[]
    for c in cuts:
        seq=np.array([remap[bid,c['kind']] for bid in c['owner']],np.int64)
        new[c['start']:c['start']+c['count']]=np.repeat(seq,3);pairs.append(seq)
    np.testing.assert_array_equal(new[:opaque_vertices],ids[:opaque_vertices]);assert np.all(new.reshape(-1,3)==new.reshape(-1,3)[:,:1])
    return (v,u,new,t),np.concatenate(pairs).astype(np.uint16)


def validate_metadata(meta,ops,table,cells,profiles,source_ops,source_meta):
    np.testing.assert_array_equal(ops,source_ops)
    rooms=Counter();entries=Counter();states=Counter();paired=0;walls=0
    for r in table:
        if r.get('role')!='wall':continue
        j=r['id'];p=profiles[r['building']];assert int(meta[j,8,0])==p['use'];assert int(meta[j,8,1])==p['glass']
        start,bays,floors,_=map(int,meta[j,1]);walls+=1
        for k in range(bays*floors):
            code=int(cells[r['cell_start']+k]);row=ops[start+k];role=code&3
            if role:
                assert int(row[2,0])==role and int(row[2,3])==1+(code>>2)
                rooms['office' if p['use']==2 else 'residential']+=1;states[int(row[2,3])]+=1
                if row[2,1]>0:assert role==2 and row[2,1]==1;paired+=1
            else:assert row[2,0]==0
        entry=ops[start+bays*floors,2];role=int(entry[0])
        if role:
            entries[{3:'lobby_glass',4:'apartment_metal',6:'house_wood'}[role]]+=1
    assert paired==300 and walls==150 and sum(entries.values())==36
    assert rooms['office']>0 and rooms['residential']>0 and entries['lobby_glass']>0
    # Geometry/slot/host transforms preserved. Only material layer, glazing
    # panel ratio and extra appearance columns may differ from stage 26.
    np.testing.assert_array_equal(meta[:len(source_meta),0,0],source_meta[:,0,0]);np.testing.assert_array_equal(meta[:len(source_meta),2:8],source_meta[:,2:8])
    return dict(rooms=dict(rooms),entry_types=dict(entries),night_states=dict(states),paired=paired,walls=walls)


def materials(source,out):
    root=Path('artifacts/l4-material-uv');assets=Path('assets/third_party')
    arrays,info=gpu.material_arrays(root,assets/'panelka',source,assets/'window_research')
    # Stage-26 extensions, keep original layer IDs and source provenance.
    for name in ('metal_roof_red','ceramic_roof','wood_door'):
        p=root/name
        arrays[0]=np.concatenate((arrays[0],gpu.linear(gpu.rgba(p/'basecolor.png',512))[None]))
        arrays[1]=np.concatenate((arrays[1],gpu.rgb(p/'normal.png',512)[None]))
        arrays[2]=np.concatenate((arrays[2],gpu.rgb(p/'orm.png',512)[None]))
    assert len(arrays[0])==16
    bank=ap.add_materials(arrays,root,out);dump(out/'material_bank.json',bank)
    extra=np.array([ap.railing(k) for k in (3,4,5)],np.float32);assert arrays[3].shape==(3,1024,1024,4)
    arrays[3]=np.concatenate((arrays[3],extra),axis=0);mips,coverage=ap.mask_mips(arrays[3])
    sheet=Image.new('RGB',(6*256,284),(228,231,232));d=ImageDraw.Draw(sheet);mask_dir=out/'mask_bank';mask_dir.mkdir(exist_ok=True)
    for i,arr in enumerate(arrays[3]):
        rgba=np.dstack((ap.srgb(arr[:,:,:3]),arr[:,:,3]));im=Image.fromarray(np.rint(rgba*255).astype(np.uint8));tile=im.resize((256,256));sheet.paste(tile,(i*256,0),tile);d.text((i*256+5,260),('roof','legacy balcony','antenna','pickets','horizontal','cross')[i],fill='black')
        if i>=3:im.save(mask_dir/('mask_'+str(i)+'.png'))
    sheet.save(out/'railing_contact.png');dump(out/'alpha_coverage.json',coverage)
    info['shared_new_materials']=len(bank);info['shared_total_materials']=len(arrays[0]);info['cutout_layers']=len(arrays[3]);info['renderer']=gpu.gl('glGetString',gpu.B,gpu.U)(0x1F01).decode()
    return arrays,mips,info,coverage


def optical_probe(renderer):
    x=np.linspace(0,1,33,dtype=np.float32);expect=np.stack([ap.glass_weights(x,*p[:2]) for p in ap.GLASS])
    assert np.all(expect>=0) and np.all(expect.sum(2)<=1.000001) and np.all(np.diff(expect[:,:,0],axis=1)<=1e-6)
    assert np.all(np.diff(expect[:,:,1],axis=1)>=-1e-6) and expect[2,-1,1]<expect[0,-1,1]
    # Actual shader execution, float FBO; same helper and constants as render.
    vert='#version 330 core\nvoid main(){vec2 p=gl_VertexID==0?vec2(-1,-1):gl_VertexID==1?vec2(3,-1):vec2(-1,3);gl_Position=vec4(p,0,1);}'
    gl,C,I,U,P,F=gpu.gl,gpu.C,gpu.I,gpu.U,gpu.P,gpu.F
    p=gl('glCreateProgram',U)()
    for typ,src in ((0x8B31,vert),(0x8B30,gpu.fragment_source(photoreal=True,probe=True))):gl('glAttachShader',None,U,U)(p,gpu.shader(typ,src))
    gl('glLinkProgram',None,U)(p);ok=I();gl('glGetProgramiv',None,U,U,P)(p,0x8B82,C.byref(ok));assert ok.value
    fb,_,_=renderer.framebuffer(33,3);gl('glUseProgram',None,U)(p);gl('glBindFramebuffer',None,U,U)(0x8D40,fb);gl('glViewport',None,I,I,I,I)(0,0,33,3)
    gl('glDisable',None,U)(0x0B71);gl('glBindVertexArray',None,U)(renderer.vao);gl('glDrawArrays',None,U,I,I)(4,0,3)
    a=np.empty((3,33,4),np.float32);gl('glReadPixels',None,I,I,I,I,U,U,P)(0,0,33,3,0x1908,0x1406,gpu.ptr(a))
    assert gl('glGetError',U)()==0 and np.isfinite(a).all();np.testing.assert_allclose(a[:,:,:2],expect,atol=.001,rtol=0);np.testing.assert_allclose(a[:,:,2],np.broadcast_to(x,(3,33)),atol=.001);assert np.all(a[:,:,3]==1.)
    gl('glUseProgram',None,U)(renderer.p);gl('glEnable',None,U)(0x0B71);gl('glDeleteProgram',None,U)(p)
    return dict(samples=99,max_gpu_error=float(np.max(abs(a[:,:,:2]-expect))),normal_incidence=expect[:,-1].tolist(),grazing=expect[:,0].tolist(),max_weight_sum=float(expect.sum(2).max()))


def render(source,out,near,lite,near_owner,lite_owner,meta,ops,records,prior):
    arrays,mips,info,coverage=materials(source,out);limit=gpu.I();gpu.gl('glGetIntegerv',None,gpu.U,gpu.P)(0x0D33,gpu.C.byref(limit));assert len(meta)<=limit.value and len(ops)<=limit.value
    vn=gpu.vertex_data(near);vl=gpu.vertex_data(lite);renderer=gpu.GL(vn,meta,ops,arrays,photoreal=True,cutout_mips=mips);del arrays,mips
    probe=optical_probe(renderer);rows=[];masks={};by_id={r['id']:r['footprint']['building_id'] for r in records}
    def draw(name,verts,center,ppm,view,mode,res,baseline=None):
        renderer.replace_vertices(verts);p=verts[:,:3];renderer.shadow_center=(p.min(0)+p.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(p,axis=0))*1.2))
        im,metrics=renderer.render(center,ppm,view,mode,tuple(res),info);assert im.size==tuple(res);im.save(out/(name+'.png'));mask=renderer.last_mask.copy()
        row=dict(name=name,mode=mode,resolution=list(res),center=list(center),ppm=ppm,view=list(view),**metrics)
        if baseline is not None:
            old=np.asarray(Image.open(source/(baseline['name']+'.png')).convert('RGB'));new=np.asarray(im);assert old.shape==new.shape
            row['changed_pixels']=int(np.any(old!=new,axis=2).sum());row['prior_foreground_pixels']=baseline['foreground_pixels'];assert row['changed_pixels']>0
        rows.append(row);return mask
    for row in prior['views']:
        name=row['name'];mode=row['mode'];verts=vn
        if name=='neighborhood_lite':verts=vl
        elif name!='neighborhood_near':
            key=next(k for k in by_id if name.startswith(k+'_'));verts=vn[near_owner==by_id[key]]
        mask=draw(name,verts,row['center'],row['ppm'],row['view'],mode,row['resolution'],row)
        if name.endswith('_detail') or name.endswith('_detail_night') or name.startswith('neighborhood_'):masks[name]=mask
    for name in ('panel_slab_a','brick_walkup_a','glass_grid_a'):np.testing.assert_array_equal(masks[name+'_detail'],masks[name+'_detail_night'])
    a,b=masks['neighborhood_near'],masks['neighborhood_lite'];iou=float(np.count_nonzero(a&b)/np.count_nonzero(a|b));assert iou>=.95
    # Additional matched angle probes on an actual upper office opening.
    bid=by_id['glass_grid_a'];v,u,ids,table=near
    walls=[r for r in table if r.get('building')==bid and r.get('role')=='wall' and r['bays']>0]
    r=max(walls,key=lambda x:(meta[x['id'],5,1]<-.1,x['length']));j=r['id'];A=np.linalg.lstsq(np.c_[u[ids==j],np.ones(np.count_nonzero(ids==j))],v[ids==j],rcond=None)[0]
    start,bays,floors,_=map(int,meta[j,1]);idx=next(start+f*bays+k for f in range(1,floors) for k in range(bays) if ops[start+f*bays+k,2,0]==1)
    q=ops[idx,0];c=np.r_[(q[:2]+q[2:])/2,1.]@A;normal=meta[j,5,:3];t=meta[j,4,:3];verts=vn[near_owner==bid]
    for name,view,mode in (('office_upper_front',normal+np.array([0,0,.03]),'day'),('office_upper_oblique',normal+.88*t+np.array([0,0,.08]),'day'),('office_upper_night',normal+np.array([0,0,.03]),'night')):
        masks[name]=draw(name,verts,c.tolist(),180.,view.tolist(),mode,(1280,960))
    np.testing.assert_array_equal(masks['office_upper_front'],masks['office_upper_night'])
    return rows,info,probe,iou,coverage


def sheets(source,out,records,labels=('26','27')):
    def compare(keys,name,cols,w,h):
        n=math.ceil(len(keys)/cols);sheet=Image.new('RGB',(cols*w,n*(h+22)*2),(235,237,239));d=ImageDraw.Draw(sheet)
        for i,k in enumerate(keys):
            x=i%cols*w;y=i//cols*(h+22)*2
            for j,root in enumerate((source,out)):
                im=Image.open(root/(k+'.png')).convert('RGB');im.thumbnail((w,h));Y=y+j*(h+22);sheet.paste(im,(x+(w-im.width)//2,Y));d.text((x+4,Y+h+3),k+' | '+labels[j],fill='black')
        sheet.save(out/(name+'.png'))
    ids=[r['id'] for r in records]
    for group in ('a','b','auto'):
        keys=[k+'_iso_gpu' for k in ids if k.endswith('_'+group)] if group!='auto' else [k+'_iso_gpu' for k in ids if k.startswith('auto_')]
        compare(keys,'comparison_'+group,5 if group!='auto' else 3,320,240)
    compare(['panel_slab_a_detail','brick_walkup_a_detail','glass_grid_a_detail','baltic_timber_a_roof','panel_slab_a_detail_night','glass_grid_a_detail_night'],'comparison_details',3,480,360)


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((SOURCE/'bundle.json').read_text())
    for k,d in manifest.items():assert sha(SOURCE/k)==d
    for name in FILES:shutil.copyfile(SOURCE/name,OUT/name);assert sha(OUT/name)==manifest[name]
    records=json.loads((SOURCE/'plans.json').read_text());placements=json.loads((SOURCE/'placements.json').read_text());table=json.loads((SOURCE/'parameters.json').read_text());prior=json.loads((SOURCE/'report.json').read_text())
    profiles={r['footprint']['building_id']:ap.profile(r) for r in records};assert profiles=={r['footprint']['building_id']:ap.profile(r) for r in reversed(records)} and len(profiles)==36
    bounds=envelopes(records,placements);near,no,cn,nv,ncheck=load_scene(SOURCE,'tile_opaque.glb',table,bounds);lite,lo,cl,lv,lcheck=load_scene(SOURCE,'tile_lite_opaque.glb',table,bounds)
    assert ncheck==prior['near_triangles']==22148 and lcheck==prior['lite_triangles']==13838
    np.testing.assert_array_equal(near[0][nv:],lite[0][lv:]);np.testing.assert_array_equal(near[1][nv:],lite[1][lv:])
    cutout_pairs={(bid,c['kind']) for c in cn for bid in c['owner']};old=np.load(SOURCE/'gpu_metadata.npz');source_meta=old['metadata'];ops=old['openings'];assert len(source_meta)==len(table)+3
    meta,remap=ap.augment(source_meta,table,profiles,cutout_pairs);near,face_ids=remap_scene(near,cn,remap,nv);lite,face_ids_lite=remap_scene(lite,cl,remap,lv);np.testing.assert_array_equal(face_ids,face_ids_lite)
    cells=np.frombuffer((SOURCE/'facade_cells.bin').read_bytes(),np.uint8);checks=validate_metadata(meta,ops,table,cells,profiles,old['openings'],source_meta)
    dump(OUT/'appearance_profiles.json',profiles);dump(OUT/'cutout_records.json',[dict(building=bid,kind=kind,id=j) for (bid,kind),j in sorted(remap.items())]);np.savez_compressed(OUT/'gpu_metadata.npz',metadata=meta,openings=ops,cutout_face_records=face_ids)
    rows,info,probe,iou,coverage=render(SOURCE,OUT,near,lite,no,lo,meta,ops,records,prior);sheets(SOURCE,OUT,records)
    counts=Counter(p['body'] for p in profiles.values());roof=Counter(p['roof'] for p in profiles.values());cut_variants=Counter(profiles[bid]['rail'] for bid,kind in remap if kind=='balcony')
    report=dict(schema='pcg.research.photoreal-appearance.v1',buildings=36,archetypes=len({p['archetype'] for p in profiles.values()}),near_triangles=ncheck,lite_triangles=lcheck,
        host_triangles=prior['host_triangles'],max_building_triangles=prior['max_building_triangles'],additional_triangles=0,source_files_identical={name:manifest[name] for name in FILES},
        source_manifest_verified=len(manifest),canonical_cells_identical=True,opening_metadata_identical=True,geometry_ownership_checks=ncheck+lcheck,
        profile_order_invariant=True,body_variants=dict(counts),roof_variants=dict(roof),cutout_variants=dict(cut_variants),cutout_records=len(remap),appearance_rows=len(meta),
        metadata_checks=checks,optical_probe=probe,alpha_coverage=coverage,near_lite_iou=iou,day_night_mask_exact=True,materials=info,views=rows,
        limitations=['Research renderer EGL llvmpipe, not native UNIGINE or measured hardware performance.','Regional palettes and aging are authored priors, not learned from real GIS buildings.','Office/lobby analytic room rays; residential approved atlas only on rear face. Virtual rooms are not actual geometry or true global illumination.','Effective dielectric/coated glass and approximate shared horizon, not measured spectral transmission or actual scene reflections.','Balcony rails and entrances vary in appearance; stage-26 geometry is unchanged. Native material/sidecar adapter still needed.','Alpha coverage correction is global per rail layer, tiny mips quantize; antenna mips remain unscaled.','No private HDR/EXR assets or new external material sources.'])
    assert report['archetypes']==15 and len(rows)==len(prior['views'])+3 and len(counts)>=10 and len(roof)>=5 and len(cut_variants)>=3
    dump(OUT/'report.json',report);dump(OUT/'bundle.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='bundle.json'})
    print('STAGE27',json.dumps({k:v for k,v in report.items() if k not in ('views','source_files_identical','alpha_coverage','limitations')}),flush=True)


if __name__=='__main__':main()
