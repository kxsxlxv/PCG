"""Stage 26 integration: stage-25 canonical plans + actual detail geometry,
shared alpha-test batches, PCGF v1 and explicit paired opening GPU sidecar.
Renders actual reloaded GLBs, using the existing software EGL PBR lab.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
from pcg.archetypes import ARCHETYPES, REGIONS
import unified_detail as detail
import compile_zero_geometry_tile as tile
from compile_zero_tiles import pack_table, LAYERS, new_float32_audit
import zero_combined_lod as combined
import zero_roof_construction as construction
import zero_alpha_cutout as cut
import zero_antenna_cards as antennas
from zero_window_entrance_stage import retain
from zero_integrated_cutouts import append_layer

sys.path.insert(0,str(Path(__file__).resolve().parent/'material_lab'))
import scene_io
import unified_metadata
import gpu_lab as gpu

SOURCE=Path('artifacts/zero-archetypes')
OUT=Path('artifacts/zero-unified-architecture')


def dump(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cases_and_plans(records):
    out=[];plans={}
    for r in sorted(records,key=lambda q:q['id']):
        q=detail.plan(r,construction.swept_pipe);v=q['variant'];f=q['footprint'];bid=f.building_id
        out.append(dict(id=r['id'],building_id=bid,outer=f.exterior,holes=f.holes,profile=v.archetype.use,
            floors=len(v.plan.floor_levels)-1,eave_m=v.eave_m,roof=v.mass.actual_roof,rise=v.top_m-v.eave_m,
            offset=r['offset'],variant=v))
        plans[bid]=q
    # Deterministic complete accessory plan, independent of input ordering.
    for r in reversed(records):
        q=detail.plan(r,construction.swept_pipe);p=plans[q['footprint'].building_id]
        assert (q['pairs'],q['rejected'],q['near'],q['lite'])==(p['pairs'],p['rejected'],p['near'],p['lite'])
        assert len(q['modules'])==len(p['modules'])
        for a,b in zip(q['modules'],p['modules']):
            assert (a.id,a.kind,a.cuts,a.antenna,a.primary,a.anchor)==(b.id,b.kind,b.cuts,b.antenna,b.primary,b.anchor)
            assert len(a.geometry)==len(b.geometry)
            for (k,m,t),(K,M,T) in zip(a.geometry,b.geometry):
                assert (k,m)==(K,M);np.testing.assert_array_equal(t,T)
    return out,plans


def audit_prior(groups,table,cells):
    prior=json.loads((SOURCE/'parameters.json').read_text())
    assert prior[:len(table)]==table
    assert (SOURCE/'facade_cells.bin').read_bytes()==cells.tobytes()
    old=[];seen=set()
    for p,a,f,_ in scene_io.glb(SOURCE/'tile.glb'):
        role=p['extras']['pcg_role'];g=groups[role];seen.add(role)
        v=scene_io.z_up(a['POSITION']);old.extend(v[f]);n=len(g['v']);nf=g['f'].size
        np.testing.assert_array_equal(v[:n],g['v'])
        np.testing.assert_array_equal(a['TEXCOORD_1'][:n],g['uv1'])
        np.testing.assert_array_equal(a['COLOR_0'][:n],g['color'])
        np.testing.assert_array_equal(f[:nf],g['f'].ravel())
    assert seen==set(tile.GROUPS)
    return np.asarray(old,np.float32)


def sidecar(plans,table,cells):
    walls={r['facade_id']:r for r in table if r.get('role')=='wall'};pairs=defaultdict(dict)
    for q in plans.values():
        for oid,p in q['pairs'].items():
            fid,tail=oid.rsplit(':window:',1);floor,bay=map(int,tail.split(':'));r=walls[fid]
            slot=floor*r['bays']+bay;assert 0<=slot<r['floors']*r['bays']
            code=int(cells[r['cell_start']+slot]);assert code&3==2
            d,w=p['door'],p['window'];left,right=sorted((d,w),key=lambda a:a[0]);post=(left[2],w[1],right[0],w[3])
            assert post[2]>post[0] and abs(d[1]-p['slab_top'])<1e-8
            pairs[str(r['id'])][str(slot)]=dict(p,code=code,post=post)
    return dict(pairs)


def export_details(groups,table,cases,plans):
    cuts=defaultdict(list);owners=defaultdict(list);manifest=[];lite_dead=set();module_counts=Counter()
    for c in cases:
        bid=c['building_id'];off=np.asarray(c['offset'])
        for m in plans[bid]['modules']:
            first=len(table)
            if m.geometry:
                combined.append_compact(groups,table,c,dict(id=m.id,kind=m.kind,geometry=m.geometry))
            if not m.primary:lite_dead.update(range(first,len(table)))
            points=[t.reshape(-1,3)+off for _,_,t in m.geometry]
            for kind,path,h,closed in m.cuts:
                assert m.primary
                v,u,f,_=cut.strip(np.asarray(path)+off,h,cut.PERIOD[kind],closed)
                cuts[kind].append((v,u,f));owners[kind].extend([bid]*f.size);points.append(v)
            if m.antenna:
                assert m.primary
                anchor,d,k=m.antenna;v,u,f=antennas.quad(np.asarray(anchor)+off,d,k)
                cuts['antenna'].append((v,u,f));owners['antenna'].extend([bid]*f.size);points.append(v)
            pts=np.concatenate(points);module_counts[m.kind]+=1
            manifest.append(dict(id=m.id,building=bid,kind=m.kind,anchor=m.anchor,primary=m.primary,
                triangles=m.triangles,bounds_world_local=[pts.min(0).tolist(),pts.max(0).tolist()]))
    assert set(cuts)=={'roof','balcony','antenna'}
    meshes={k:cut.join(cuts[k]) for k in ('roof','balcony','antenna')}
    images={k:cut.pattern(k,512) for k in ('roof','balcony')};images['antenna']=antennas.atlas()
    actual={}
    for name,kinds in (('roof_balcony_cutout.glb',('roof','balcony')),('antenna_cutout.glb',('antenna',))):
        g={k:meshes[k] for k in kinds};im={k:images[k] for k in kinds};p=OUT/name
        cut.export(p,g,im);actual.update(cut.read(p,g,im))
        # GPU material array layer order is the GLB primitive order.
        assert [x[3]['materials'][x[0]['material']]['name'] for x in scene_io.glb(p)]==['cutout_'+k for k in kinds]
    return actual,owners,manifest,lite_dead,module_counts


def scene_from(groups,table,path,actual,cut_owners):
    tile.export_glb(groups,path);v,u,ids,_=tile.read_glb(path,groups,table)
    owner=np.array([table[int(i)]['building'] for i in ids]);scene=(v,u,ids,list(table))
    for k in ('roof','balcony','antenna'):
        vv,uu,ff,_=actual[k];scene=append_layer(scene,vv,uu,ff,k)
        owner=np.concatenate((owner,np.array(cut_owners[k])))
    assert len(owner)==len(scene[0]) and np.all(owner.reshape(-1,3)==owner.reshape(-1,3)[:,:1])
    assert np.all(np.linalg.norm(np.cross(scene[0][1::3]-scene[0][::3],scene[0][2::3]-scene[0][::3]),axis=1)>1e-9)
    return scene,owner


def audit_gpu_lookup(table,meta,ops,pairs):
    """Independent explicit FP32 rectangle oracle vs the shader slot lookup.
    This validates the uploaded metadata; it is not a GPU execution timing.
    """
    samples=0;max_drift=0.
    for r in table:
        if r.get('role')!='wall':continue
        j=r['id'];start,bays,floors,_=map(int,meta[j,1]);center,spacing,h=meta[j,2,:3]
        entry=start+bays*floors;rows=ops[start:entry+1];rng=np.random.default_rng(r['seed']^0x26)
        s=(rng.random((1024,2))*[r['length'],r['height']]).astype(np.float32)
        # Exact rectangle center samples ensure every aperture/paired half is exercised.
        points=[(q[:2]+q[2:])/2 for row in rows for q in row[:2] if q[2]>q[0] and q[3]>q[1]]
        if points:s=np.vstack((s,np.asarray(points,np.float32)))
        expect=np.zeros((len(s),3),np.float32);got=expect.copy()
        for row in rows:
            for k,rect in enumerate(row[:2]):
                m=np.all(s>=rect[:2],axis=1)&np.all(s<rect[2:],axis=1)
                assert not np.any(expect[m,0]);expect[m]=[row[2,k],row[2,2],row[2,3]]
        floor=np.floor(s[:,1]/h).astype(int);bay=np.floor((s[:,0]-center)/spacing+np.float32(.5)).astype(int)
        valid=(floor>=0)&(floor<floors)&(bay>=0)&(bay<bays);idx=np.flatnonzero(valid)
        if len(idx):
            q=ops[start+floor[idx]*bays+bay[idx]];p=s[idx]
            for k in (0,1):
                hit=np.all(p>=q[:,k,:2],axis=1)&np.all(p<q[:,k,2:],axis=1)
                got[idx[hit]]=np.c_[q[hit,2,k],q[hit,2,2:4]]
        en=ops[entry];m=np.all(s>=en[0,:2],axis=1)&np.all(s<en[0,2:],axis=1)
        assert not np.any(got[m,0]);got[m]=en[2,[0,2,3]]
        np.testing.assert_array_equal(got,expect);samples+=len(s)
        for slot,p in pairs.get(str(j),{}).items():
            for k,name in enumerate(('door','window')):
                max_drift=max(max_drift,float(np.max(abs(np.asarray(p[name])-ops[start+int(slot),k]))))
    assert max_drift<=1e-4
    return dict(samples=samples,max_paired_edge_drift_m=max_drift)


def material_arrays():
    root=Path('artifacts/l4-material-uv');assets=Path('assets/third_party')
    arrays,info=gpu.material_arrays(root,assets/'panelka',OUT,assets/'window_research')
    # Add original shared metric recipes; no per-building maps or private assets.
    for family in ('metal_roof_red','ceramic_roof','wood_door'):
        p=root/family
        arrays[0]=np.concatenate((arrays[0],gpu.linear(gpu.rgba(p/'basecolor.png',512))[None]))
        arrays[1]=np.concatenate((arrays[1],gpu.rgb(p/'normal.png',512)[None]))
        arrays[2]=np.concatenate((arrays[2],gpu.rgb(p/'orm.png',512)[None]))
    assert len(arrays[0])==len(arrays[1])==len(arrays[2])==16
    info['extra_material_families']=['metal_roof_red','ceramic_roof','wood_door']
    info['renderer']=gpu.gl('glGetString',gpu.B,gpu.U)(0x1F01).decode();return arrays,info


def render(near,lite,near_owner,lite_owner,meta,ops,records,plans,old_vertices):
    limit=gpu.I();gpu.gl('glGetIntegerv',None,gpu.U,gpu.P)(0x0D33,gpu.C.byref(limit))
    assert max(len(meta),len(ops))<=limit.value
    arrays,info=material_arrays();vn=gpu.vertex_data(near);vl=gpu.vertex_data(lite);renderer=gpu.GL(vn,meta,ops,arrays);del arrays
    rows=[];images={}
    def fit(points,view,res,center=None):
        points=np.asarray(points,float);d=gpu.norm(view);right=gpu.norm(np.cross([0,0,1],d));up=np.cross(d,right)
        if center is None:center=(points.min(0)+points.max(0))/2
        p=points-center;W,H=res;span=max(np.ptp(p@up),np.ptp(p@right)*H/W)
        return center,H*.78/max(span,1.)
    def draw(name,verts,center,ppm,view,mode,res):
        renderer.replace_vertices(verts);p=verts[:,:3];renderer.shadow_center=(p.min(0)+p.max(0))/2
        renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(p,axis=0))*1.2))
        im,metrics=renderer.render(center,ppm,view,mode,res,info)
        assert im.size==res;im.save(OUT/(name+'.png'));rows.append(dict(name=name,mode=mode,resolution=list(res),center=np.asarray(center).tolist(),ppm=float(ppm),view=list(view),**metrics))
        return im,renderer.last_mask.copy()
    by_id={r['id']:r for r in records};order=[a+'_'+k for k in ('a','b') for a in ARCHETYPES]+['auto_'+r for r in REGIONS]
    for key in order:
        r=by_id[key];bid=r['footprint']['building_id'];q=plans[bid];v=q['variant'];off=np.asarray(r['offset']);host=np.asarray(v.mass.shell.vertices)+off
        lo,hi=host.min(0),host.max(0);bbox=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]);view=(1.,-1.5,.85);res=(1280,960);c,ppm=fit(bbox,view,res)
        im,_=draw(key+'_iso_gpu',vn[near_owner==bid],c,ppm,view,'day',res);images[key]=im
    for kind in ('a','b','auto'):
        keys=[a+'_'+kind for a in ARCHETYPES] if kind!='auto' else ['auto_'+r for r in REGIONS]
        cols=5 if kind!='auto' else 3;sheet=Image.new('RGB',(cols*320,math.ceil(len(keys)/cols)*264),(239,241,242));d=ImageDraw.Draw(sheet)
        for i,key in enumerate(keys):
            im=images[key].copy();im.thumbnail((320,240));x,y=(i%cols)*320,(i//cols)*264;sheet.paste(im,(x+(320-im.width)//2,y));d.text((x+5,y+242),key,fill='black')
        sheet.save(OUT/('contact_'+kind+'.png'))
    comparison=Image.new('RGB',(1600,6*264),(239,241,242));d=ImageDraw.Draw(comparison)
    for i,a in enumerate(ARCHETYPES):
        key=a+'_a';x=(i%5)*320;y=(i//5)*528
        for j,im in enumerate((Image.open(SOURCE/(key+'_iso.png')),images[key])):
            im=im.copy();im.thumbnail((320,240));yy=y+j*264;comparison.paste(im,(x+(320-im.width)//2,yy));d.text((x+4,yy+242),key+(' CPU 25' if j==0 else ' GPU 26'),fill='black')
    comparison.save(OUT/'comparison_a.png')
    # Matched close day/night and roof views; stage-25 host is the same camera anchor.
    for key in ('panel_slab_a','brick_walkup_a','baltic_timber_a','glass_grid_a'):
        r=by_id[key];bid=r['footprint']['building_id'];v=plans[bid]['variant'];off=np.asarray(r['offset']);wall=max([w for w in v.plan.facades if not w.party and not w.courtyard],key=lambda w:(w.outward[1]<-.1,w.length))
        op=next((o for o in wall.openings if o.role=='balcony_door' and o.floor>=1),None) or next((o for o in wall.openings if o.role=='door'),None) or wall.openings[0]
        c=np.asarray(wall.point(op.u+op.width/2,op.z+op.height/2,-.25))+off;t=(np.array(wall.end)-wall.start)/wall.length;n=np.array(wall.outward);view=(*tuple(n+.35*t),.18);res=(1280,960);verts=vn[near_owner==bid]
        _,mask=draw(key+'_detail',verts,c,125.,view,'day',res)
        if key in ('panel_slab_a','brick_walkup_a','glass_grid_a'):
            _,night=draw(key+'_detail_night',verts,c,125.,view,'night',res);np.testing.assert_array_equal(mask,night)
        if key in ('panel_slab_a','baltic_timber_a'):
            host=np.asarray(v.mass.shell.vertices)+off;c=(host.min(0)+host.max(0))/2;c[2]=v.top_m+off[2];view=(1.,-1.5,1.65);res=(1280,960);_,ppm=fit(host,view,res,c)
            draw(key+'_roof',verts,c,ppm*1.15,view,'day',res)
    res=(1920,1080);view=(1.,-1.6,1.3);c,ppm=fit(old_vertices,view,res)
    _,a=draw('neighborhood_near',vn,c,ppm,view,'day',res);_,b=draw('neighborhood_lite',vl,c,ppm,view,'day',res)
    iou=float(np.count_nonzero(a&b)/np.count_nonzero(a|b));assert iou>=.95,iou
    return rows,info,iou,limit.value


def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    source_hash=json.loads((SOURCE/'bundle.json').read_text())
    for name,digest in source_hash.items():assert sha(SOURCE/name)==digest
    records=json.loads((SOURCE/'plans.json').read_text());cases,plans=cases_and_plans(records)
    groups,table,cells,hosts=tile.compile_scene(cases,audit_samples=1024);old_vertices=audit_prior(groups,table,cells)
    host_n=sum(len(g['f']) for g in groups.values());host_records=len(table);pairs=sidecar(plans,table,cells)
    actual,owners,placements,dead,counts=export_details(groups,table,cases,plans)
    near,no=scene_from(groups,table,OUT/'tile_opaque.glb',actual,owners)
    light_groups,removed=retain(groups,dead);lite,lo=scene_from(light_groups,table,OUT/'tile_lite_opaque.glb',actual,owners)
    assert near[3]==lite[3] and len(near[0])-len(lite[0])==removed*3
    for c in cases:
        bid=c['building_id'];q=plans[bid]
        assert np.count_nonzero(no==bid)//3==q['near'] and np.count_nonzero(lo==bid)//3==q['lite']
    (OUT/'facade_cells.bin').write_bytes(cells.tobytes());audit=new_float32_audit();samples=pack_table(table,cells,OUT/'facade_params.bin',audit)
    cpp=json.loads(subprocess.run([str(decoder),str(OUT/'facade_params.bin'),str(OUT/'facade_cells.bin'),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert cpp['records']==len(table) and cpp['cells']==len(cells)
    dump(OUT/'parameters.json',table);dump(OUT/'opening_blocks_v2.json',dict(schema='pcg.window-blocks.v2-research',legacy_cells_unchanged=True,paired_openings=pairs));dump(OUT/'placements.json',placements)
    dump(OUT/'plans.json',records)
    styles={r['footprint']['building_id']:r['decision']['archetype'] for r in records}
    M,O,models,metadata=unified_metadata.build(near,cells,pairs,styles)
    assert len(models)==counts['balcony_glazed'] and counts['balcony_glazed']+counts['balcony_open']==300
    lookup=audit_gpu_lookup(table,M,O,pairs);np.savez_compressed(OUT/'gpu_metadata.npz',metadata=M,openings=O)
    rows,info,iou,limit=render(near,lite,no,lo,M,O,records,plans,old_vertices)
    buildings=[dict(building=bid,archetype=q['variant'].archetype.id,roof=q['variant'].mass.actual_roof,near=q['near'],lite=q['lite'],
        source_top=q['variant'].top_m,plan_digest=q['variant'].plan.digest(),paired=len(q['pairs']),rejected=q['rejected']) for bid,q in plans.items()]
    report=dict(schema='pcg.research.unified-architecture.v1',source_stage=25,buildings=len(buildings),archetypes=len({b['archetype'] for b in buildings}),host_triangles=host_n,
        host_records=host_records,near_triangles=len(near[0])//3,lite_triangles=len(lite[0])//3,max_building_triangles=max(b['near'] for b in buildings),
        full_budget=detail.CAP,opening_triangles_added=0,counts=dict(counts),paired_openings=metadata['paired'],glazed_models=len(models),
        envelope_checks=sum(q['checks'] for q in plans.values()),legacy_abi_samples=samples,legacy_abi_audit=audit,cpp_abi=cpp,gpu_lookup=lookup,gpu_metadata=metadata,
        max_texture_size=limit,source_host_exact=True,canonical_cells_identical=True,glb_roundtrip=True,input_order_invariant=True,alpha_test=True,
        lite_overview_silhouette_iou=iou,material_sources=info,buildings_detail=buildings,views=rows,limitations=[
        'Synthetic stage-25 inputs and regional priors; no production GIS validation.',
        'SDK-free EGL software renderer: same PBR/POM/virtual room and alpha test paths, not native UNIGINE or GPU hardware timing.',
        'Paired window/door rectangles use an explicit research sidecar; native ABI v2 remains to implement.',
        'Lite removes complete minor detail modules; not a measured native far-LOD/SSE threshold.',
        'Source host top/footprint exact; architectural roof equipment and overhangs intentionally extend render envelope.',
        'Timber/cladding and curtain wall remain shader proxies; dormer front uses a window mask on actual geometry. Private EXR/HDR excluded.',
        'Accessory envelope checks are conservative; no structural engineering certification, general concave pitched roof or city-scale adjacency test.'])
    assert report['buildings']==36 and report['archetypes']==15 and report['max_building_triangles']<=detail.CAP
    dump(OUT/'report.json',report);dump(OUT/'bundle.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='bundle.json'})
    print('STAGE26',json.dumps({k:v for k,v in report.items() if k not in ('buildings_detail','views','legacy_abi_audit','limitations')}),flush=True)


if __name__=='__main__':main(sys.argv[1])
