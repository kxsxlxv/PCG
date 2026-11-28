"""Stage 30 opt-in section grammar on actual exported geometry and shared PBR.
Synthetic matched inputs only in CI. Private Stage-29 overlay is offline only.
"""
from collections import Counter
from dataclasses import asdict
import argparse, hashlib, json, math, os, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pcg.footprint import Footprint
from pcg.reference_grammar import RULES,baseline,build
from pcg.archetypes import stable
import compile_zero_geometry_tile as tile
from compile_zero_tiles import pack_table,LAYERS,new_float32_audit
import unified_detail as detail
import detail28
import equipment29
import zero_combined_lod as combined
import zero_unified_architecture as uni
from zero_roof_construction import swept_pipe
from zero_window_entrance_stage import retain
import zero_photoreal as prev
import appearance27 as ap27
import appearance28 as ap28
import external29 as ext
import unified_metadata as um
import gpu_lab as gpu
from quant_probe import quant_probe

ROOT=Path('artifacts/zero-natural-architecture')
def dump(path,obj):path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixtures(variant):
    # Authored metric inputs informed by donor aspect ratios and periodicities.
    # Not copied donor footprints, heights or identified building series.
    spec=((72.,15.,5,'flat',0.),(116.,14.,11,'flat',0.),(93.,16.,13,'flat',0.),
          (48.,20.,17,'flat',0.),(52.,18.,22,'flat',0.),(18.,14.,2,'hip',2.4),
          (38.,15.,6,'flat',0.),(60.,20.,1,'gable',2.4))
    for i,(name,(w,d,n,roof,rise)) in enumerate(zip(RULES,spec)):
        holes=[];party=()
        if variant:w*=.76+.08*(i%3);d*=1.10;n+=1 if i not in (5,7) else 0
        outer=[(0,0),(w,0),(w,d),(0,d)]
        if variant and i==2:outer=[(0,0),(w,0),(w,d),(w*.48,d),(w*.48,d*1.65),(0,d*1.65)]
        if variant and i==4:outer=[(0,0),(w*.5,0),(w*.5,d*.18),(w,d*.18),(w,d),(0,d)]
        if variant and i==6:
            holes=[[(w*.28,d*.30),(w*.72,d*.30),(w*.72,d*.70),(w*.28,d*.70)]]
        if variant and i==7:roof='flat';rise=0.
        if variant:
            a=math.radians(17 if i%2==0 else -23);R=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
            outer=(np.asarray(outer)@R.T).tolist();holes=[(np.asarray(h)@R.T).tolist() for h in holes]
        f=Footprint(f'reference30:{name}:{variant}',outer,holes)
        if variant and i==0:party=((0,1),)
        eave=n*RULES[name].pitch;top=eave+rise
        # All geometry and renderer coordinates stay within local float range.
        off=[float((i%3-1)*142),float((i//3-1)*86),0.]
        yield dict(id=f'{name}_{variant}',rule=name,footprint=f,floors=n,eave=eave,top=top,roof=roof,party=party,offset=off)

def compile_case(specs,candidate,out,decoder,calibration=None):
    out.mkdir(parents=True,exist_ok=True);uni.OUT=out
    cases=[];plans={};records=[];reference={};checks=0
    for s in sorted(specs,key=lambda x:x['id']):
        f=s['footprint'];bid=f.building_id;name=s['rule'];kw=dict(roof=s['roof'],eave=s['eave'],party_edges=s['party'])
        b=baseline(f,RULES[name],s['floors'],s['top'],**kw)
        measured=None if calibration is None else calibration.get(bid)
        v,ref=build(f,name,s['floors'],s['top'],volume=candidate in ('volume','efficient'),measured=measured,**kw) if candidate else (b,None)
        assert v.mass.shell==b.mass.shell and v.plan.footprint_digest==b.plan.footprint_digest==f.digest()
        assert v.top_m==s['top'] and v.eave_m==s['eave'] and v.mass.actual_roof==s['roof']
        assert tuple(v.plan.floor_levels)==tuple(b.plan.floor_levels)
        r=dict(id=s['id'],footprint=asdict(f),decision=v.decision(),architecture=asdict(v.plan),offset=s['offset'],source_constraints=kw)
        if ref:r['reference_grammar']=ref.canonical();reference[bid]=ref
        q=detail.plan(r,swept_pipe,(f,v),ref,efficient=candidate=='efficient');plans[bid]=q;records.append(r);checks+=q['checks']
        c=dict(id=s['id'],building_id=bid,outer=f.exterior,holes=f.holes,profile=v.archetype.use,floors=s['floors'],eave_m=v.eave_m,roof=v.mass.actual_roof,rise=v.top_m-v.eave_m,offset=s['offset'],variant=v,party_edges=s['party'])
        cases.append(c)
        # Replan the same input independently and compare every semantic intent.
        if ref:
            vv,rr=build(f,name,s['floors'],s['top'],volume=candidate in ('volume','efficient'),measured=measured,**kw);assert vv.plan.digest()==v.plan.digest() and rr==ref
        assert all(not w.openings for w in v.plan.facades if w.party)
    groups,table,cells,hosts=tile.compile_scene(cases,audit_samples=768)
    # Exact canonical host vertices and records are deterministic under input reorder.
    g2,t2,c2,_=tile.compile_scene(list(reversed(cases)),audit_samples=64)
    assert table==t2;np.testing.assert_array_equal(cells,c2)
    for role,g in groups.items():
        for k,a in g.items():np.testing.assert_array_equal(a,g2[role][k])
    host_counts={c['building_id']:len(c['variant'].mass.shell.triangles()) for c in cases}
    pairs=uni.sidecar(plans,table,cells)
    actual,owners,placements,dead,counts=uni.export_details(groups,table,cases,plans)
    walls={r['facade_id']:r for r in table if r.get('role')=='wall'}
    # Projected front uses the identical metric host U/Z frame and opening IDs.
    # Keep ordinary UV0 and PCGF record layout unchanged; front alias is a
    # research sidecar, not an incompatible modification of the legacy ABI.
    projection_hosts={};proj_by={q.id:q for ref in reference.values() for q in ref.projections}
    offs={r['footprint']['building_id']:np.array(r['offset']) for r in records}
    roof=groups['roof'];rid=roof['color'][:,0].astype(int)+256*roof['color'][:,1].astype(int)
    for r in table:
        if r.get('accessory')!='ref_projection_front':continue
        q=proj_by[r['placement']];host=walls[q.facade];w=next(w for w in plans[r['building']]['variant'].plan.facades if w.id==q.facade)
        mask=rid==r['id'];p=roof['v'][mask].astype(float)-offs[r['building']];T=(np.asarray(w.end)-w.start)/w.length;N=np.asarray(w.outward)
        u=(p[:,:2]-w.start)@T;z=p[:,2]
        np.testing.assert_allclose((p[:,:2]-w.start)@N,q.depth,atol=5e-5,rtol=0)
        assert len(p)==4 and min(u)>q.u-5e-5 and max(u)<q.U+5e-5
        uv=np.column_stack((u,z)).astype(np.float32);roof['uv1'][mask]=uv;projection_hosts[r['id']]=host['id']
    near,no=uni.scene_from(groups,table,out/'tile_opaque.glb',actual,owners)
    litegroups,_=retain(groups,dead);lite,lo=uni.scene_from(litegroups,table,out/'tile_lite_opaque.glb',actual,owners)
    assert near[3]==lite[3]
    for bid,q in plans.items():
        assert np.count_nonzero(no==bid)//3==q['near'] and np.count_nonzero(lo==bid)//3==q['lite'] and 0<q['lite']<=q['near']<=1800
    (out/'facade_cells.bin').write_bytes(cells.tobytes());audit=new_float32_audit();samples=pack_table(table,cells,out/'facade_params.bin',audit)
    cpp=json.loads(subprocess.run([str(decoder),str(out/'facade_params.bin'),str(out/'facade_cells.bin'),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert cpp['records']==len(table) and cpp['cells']==len(cells)
    styles={r['footprint']['building_id']:r['decision']['archetype'] for r in records}
    M,O,models,md=um.build(near,cells,pairs,styles)
    extra=[]
    for bid,ref in reference.items():
        for e in ref.entries:
            r=walls[e.facade];start,bays,floors,_=map(int,M[r['id'],1]);slot=start+e.bay
            assert cells[r['cell_start']+e.bay]==0 and O[slot,2,0]==0
            assert e.rect[0]>r['first_center']+r['spacing']*(e.bay-.5) and e.rect[2]<r['first_center']+r['spacing']*(e.bay+.5)
            assert e.rect[3]<r['floor_h']
            role=3 if ref.rule.use=='office' and ref.rule.id!='service_hall' else 4
            O[slot,0]=e.rect;O[slot,2]=[role,0,stable(e.id)%4096,1]
            extra.append(dict(id=e.id,facade=e.facade,record=r['id'],slot=slot,rect=e.rect,role=role))
    profiles={r['footprint']['building_id']:ap27.profile(r) for r in records}
    for bid,ref in reference.items():
        if ref.rule.use=='apartments':
            palette=((.63,.37,.28),(.36,.48,.60),(.48,.53,.33),(.55,.38,.31),(.49,.53,.57))
            profiles[bid]['panel']=list(palette[(list(RULES).index(ref.rule.id)+stable('ref-color',bid)%2)%len(palette)])
        if ref.rule.id=='service_hall':profiles[bid]['body']='panel_gray';profiles[bid]['roof']='metal_graphite' if next(r for r in records if r['footprint']['building_id']==bid)['decision']['actual_roof']!='flat' else 'membrane_gray'
    # Reuse the actual Stage 28 bounded geometry pass on new plans. It sees
    # projected columns and all secondary entries as real obstacle geometry.
    refinement_counts=Counter();refinement_checks={};equipment_report=None
    if candidate in ('volume','efficient'):
        prior={'buildings_detail':[dict(building=bid,near=q['near'],lite=q['lite']) for bid,q in plans.items()]}
        mods,stats,refinement_checks=detail28.plan(records,placements,near,O,M,profiles,prior,require_added=False,efficient=candidate=='efficient')
        again,stats2,checks2=detail28.plan(list(reversed(records)),placements,near,O,M,profiles,prior,require_added=False,efficient=candidate=='efficient')
        assert stats==stats2 and refinement_checks==checks2 and len(mods)==len(again)
        for x,y in zip(mods,again):
            assert {k:v for k,v in x.items() if k!='geometry'}=={k:v for k,v in y.items() if k!='geometry'}
            for (k,m,t),(K,N,T) in zip(x['geometry'],y['geometry']):
                assert (k,m)==(K,N);np.testing.assert_array_equal(t,T)
        old_M=M.copy();old_O=O.copy();old_count=len(table)
        for m in mods:
            bid=m['building'];v=plans[bid]['variant'];start=len(table)
            c=dict(building_id=bid,offset=[0.,0.,0.],eave_m=v.eave_m,floors=len(v.plan.floor_levels)-1)
            combined.append_compact(groups,table,c,m)
            if not m['primary']:dead.update(range(start,len(table)))
            pts=np.concatenate([q.reshape(-1,3) for _,_,q in m['geometry']])
            placements.append(dict(**{k:v for k,v in m.items() if k!='geometry'},triangles=sum(len(q) for _,_,q in m['geometry']),bounds_world_local=[pts.min(0).tolist(),pts.max(0).tolist()],record_range=[start,len(table)]))
            refinement_counts[m['kind']]+=1
        for bid,q in plans.items():
            q['near']=stats[bid]['near'];q['lite']=stats[bid]['lite']
        counts.update(refinement_counts)
        near,no=uni.scene_from(groups,table,out/'tile_opaque.glb',actual,owners)
        litegroups,_=retain(groups,dead);lite,lo=uni.scene_from(litegroups,table,out/'tile_lite_opaque.glb',actual,owners)
        assert near[3]==lite[3]
        for bid,q in plans.items():
            assert np.count_nonzero(no==bid)//3==q['near'] and np.count_nonzero(lo==bid)//3==q['lite']
        # Existing record IDs/UVs and the canonical PCGF payload stay exact;
        # new opaque records are appended before the cutout rows.
        oldabi=(out/'facade_params.bin').read_bytes()
        audit=new_float32_audit();samples=pack_table(table,cells,out/'facade_params.bin',audit)
        assert (out/'facade_params.bin').read_bytes()[16:len(oldabi)]==oldabi[16:]
        cpp=json.loads(subprocess.run([str(decoder),str(out/'facade_params.bin'),str(out/'facade_cells.bin'),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
        assert cpp['records']==len(table) and cpp['cells']==len(cells)
        M,O,models,md=um.build(near,cells,pairs,styles)
        for x in extra:
            O[x['slot'],0]=x['rect'];O[x['slot'],2]=old_O[x['slot'],2]
        np.testing.assert_array_equal(M[:old_count],old_M[:old_count]);np.testing.assert_array_equal(O,old_O)
    # Cutout rows are allocated per building and kind, same as stage 27.
    cut_pairs={(bid,k) for k,own in owners.items() for bid in set(own)}
    meta,remap=ap27.augment(M[:len(table)],table,profiles,cut_pairs)
    proxy,proxyman,proxyids=ap28.proxies(records,profiles)
    for r in table:meta[r['id'],10,3]=proxyids[r['building']]
    for (bid,k),j in remap.items():meta[j,10,3]=proxyids[bid]
    patterns={p['facade']:p for ref in reference.values() for p in ref.facades}
    for r in table:
        if r.get('role')!='wall' or r['building'] not in reference:continue
        ref=reference[r['building']];j=r['id'];style=list(RULES).index(ref.rule.id)+1
        if r['facade_id'] not in patterns:continue
        p=patterns[r['facade_id']];meta[j,13,3]=p['section_bays'];meta[j,7,1:4]=[2. if style in (2,3) else 1.,style,4. if candidate=='efficient' else 2. if candidate=='volume' else 0.]
        if style==8:
            entry=int(meta[j,1,0]+meta[j,1,1]*meta[j,1,2]);
            if O[entry,2,0]:O[entry,2,0]=4
    if candidate=='efficient':
        for r in table:
            if r.get('accessory') in ('slab','ref_core_plain','ref_core_access','ref_core_louver'):
                j=r['id'];meta[j,7,:2]=r['span'];meta[j,7,3]=4.
    lookup=uni.audit_gpu_lookup(table,M,O,pairs)
    for j,k in projection_hosts.items():
        meta[j]=meta[k];meta[j,7,3]=5. if candidate=='efficient' else 3. if candidate=='volume' else 1.
        if table[j]['family']=='concrete_panel':meta[j,0,1:3]=ap27.LAYERS['panel_white' if stable(table[j]['building'])%2 else 'panel_warm']
        # A front must share the exact same aperture range and frame as its host.
        np.testing.assert_array_equal(meta[j,1:6],meta[k,1:6]);assert meta[j,0,0]==1
    def remap_scene(s,owner):
        v,u,ids,_=s;ids=ids.copy();n=len(table)
        for k in ('roof','balcony','antenna'):
            old=n+('roof','balcony','antenna').index(k);mask=s[2]==old
            ids[mask]=[remap[(bid,k)] for bid in owner[mask]]
        assert np.all(ids.reshape(-1,3)==ids.reshape(-1,3)[:,:1])
        return (v,u,ids,table)
    near=remap_scene(near,no);lite=remap_scene(lite,lo)
    cut_faces=near[2].reshape(-1,3)[:,0];cut_faces=cut_faces[cut_faces>=len(table)].astype(np.uint16)
    lite_faces=lite[2].reshape(-1,3)[:,0];np.testing.assert_array_equal(cut_faces,lite_faces[lite_faces>=len(table)])
    assert len(cut_faces)>0
    # Front aliases are opaque geometry and are shared exactly between Near/Lite.
    for j in projection_hosts:
        np.testing.assert_array_equal(near[0][near[2]==j],lite[0][lite[2]==j]);np.testing.assert_array_equal(near[1][near[2]==j],lite[1][lite[2]==j])
    limit=gpu.I();gpu.gl('glGetIntegerv',None,gpu.U,gpu.P)(0x0D33,gpu.C.byref(limit));assert len(meta)<=limit.value and len(O)<=limit.value
    per=[]
    for s in specs:
        bid=s['footprint'].building_id;q=plans[bid];ref=reference.get(bid);mod=Counter(m.kind for m in q['modules'])
        per.append(dict(id=s['id'],building=bid,rule=s['rule'],near=q['near'],lite=q['lite'],host=host_counts[bid],floors=s['floors'],openings=sum(len(w.openings) for w in q['variant'].plan.facades),secondary_entries=len(ref.entries) if ref else 0,projection_intents=len(ref.projections) if ref else 0,projections=mod['ref_projection'],roof_core_intents=len(ref.roof_cores) if ref else 0,roof_cores=mod['ref_roof_core'],module_counts=dict(mod),rejected=q['rejected'],compact_saved=q['compact_saved']))
    for name,value in [('plans.json',records),('parameters.json',table),('placements.json',placements),('section_entries.json',extra),('projected_front_aliases.json',{str(k):v for k,v in projection_hosts.items()}),('opening_blocks_v2.json',dict(paired_openings=pairs)),('reflection_proxies.json',proxyman),('cutout_records.json',[dict(building=bid,kind=k,id=j) for (bid,k),j in sorted(remap.items())])]:dump(out/name,value)
    np.savez_compressed(out/'gpu_metadata.npz',metadata=meta,openings=O,reflection_proxies=proxy,cutout_face_records=cut_faces)
    if candidate in ('volume','efficient'):
        # Reuse Stage 29's actual directional equipment meshes and UV tags.
        # Read back the refined exported GLBs, not an analytical proxy.
        prior={'per_building':{bid:dict(near=q['near'],lite=q['lite']) for bid,q in plans.items()}}
        with tempfile.TemporaryDirectory() as tmp:
            geom=Path(tmp)/'equipment'
            equipment_report=equipment29.build(out,geom,records,placements,table,prior,require_all_variants=False,efficient=candidate=='efficient')
            rev=Path(tmp)/'reverse'
            eq2=equipment29.build(out,rev,list(reversed(records)),list(reversed(placements)),table,prior,require_all_variants=False,efficient=candidate=='efficient')
            assert eq2==equipment_report
            for name in ('tile_opaque.glb','tile_lite_opaque.glb'):
                assert sha(geom/name)==sha(rev/name)
                shutil.copyfile(geom/name,out/name)
            for name in ('placements.json','equipment_report.json'):shutil.copyfile(geom/name,out/name)
        bounds=prev.envelopes(records,placements)
        def reload(name):
            scene,owner,cuts,nv,count=prev.load_scene(out,name,table,bounds)
            scene,face=prev.remap_scene(scene,cuts,remap,nv);np.testing.assert_array_equal(face,cut_faces)
            return scene,owner,count
        near,no,nn=reload('tile_opaque.glb');lite,lo,ll=reload('tile_lite_opaque.glb')
        assert nn==equipment_report['near_triangles'] and ll==equipment_report['lite_triangles']
        for row in per:
            x=equipment_report['per_building'][row['building']]
            row['near']=x['near'];row['lite']=x['lite']
            assert np.count_nonzero(no==row['building'])//3==x['near'] and np.count_nonzero(lo==row['building'])//3==x['lite'] and 0<x['lite']<=x['near']<=1800
    report=dict(buildings=per,near_triangles=len(near[0])//3,lite_triangles=len(lite[0])//3,max_building_triangles=max(p['near'] for p in per),counts=dict(counts),metadata_rows=len(meta),opening_rows=len(O),shared_material_layers=36,footprints_and_heights_exact=True,canonical_host_deterministic=True,abi_samples=samples,cpp=cpp,gpu_lookup=lookup,placement_checks=checks,secondary_entries=len(extra),projected_fronts=len(projection_hosts),texture_limit=limit.value)
    if candidate in ('volume','efficient'):report.update(refinement28=dict(refinement_counts),refinement28_checks=refinement_checks,equipment29=equipment_report)
    dump(out/'report.json',report)
    return dict(near=near,lite=lite,no=no,lo=lo,meta=meta,ops=O,proxy=proxy,report=report,records=records)

def load_materials(out,private):
    bank=out/'material_bank_base';bank.mkdir(exist_ok=True)
    arrays,mips,info,_=prev.materials(ROOT,bank);ap28.replace_maps(arrays,bank)
    if private is not None:
        if os.getenv('GITHUB_ACTIONS')=='true':raise RuntimeError('private overlay prohibited in CI')
        report=json.loads((private/'report.json').read_text());assert report['source_kind']=='private'
        for i in ext.SLOTS:
            for k,name in enumerate(('basecolor','normal','orm')):
                p=private/'material_bank'/str(i)/(name+'.png');a=np.asarray(Image.open(p).convert('RGB'),np.float32)/255.
                if k==0:arrays[k][i,:,:,:3]=ext.linear(a);arrays[k][i,:,:,3]=1
                else:arrays[k][i]=a
    return arrays,mips,info

def run(out,decoder,private=None,volume=False,efficient=False):
    if efficient:volume=True
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'bundle.json').read_text())
    for k,h in manifest.items():assert sha(ROOT/k)==h,k
    arrays,mips,info=load_materials(out,private);allrows=[];views=[];images=[];quant=None;upgrades=[]
    for variant in (0,1):
        specs=list(fixtures(variant));legacy=compile_case(specs,'volume' if efficient else True if volume else False,out/f'legacy_{variant}',decoder);cand=compile_case(specs,'efficient' if efficient else 'volume' if volume else True,out/f'grammar_{variant}',decoder)
        if volume:upgrades.append(dict(stage28=cand['report']['refinement28'],stage29=cand['report']['equipment29']))
        if efficient:
            assert legacy['records']==cand['records'],'canonical plans or GIS changed'
            np.testing.assert_array_equal(legacy['ops'],cand['ops'])
            for old,new in zip(legacy['report']['buildings'],cand['report']['buildings']):
                assert old['id']==new['id'] and old['module_counts']==new['module_counts'] and old['openings']==new['openings']
        # Match exact host geometry for each case even when window/section metadata differs.
        for a,b in zip(legacy['records'],cand['records']):assert a['footprint']==b['footprint'] and a['decision']['top_m']==b['decision']['top_m'] and a['decision']['eave_m']==b['decision']['eave_m']
        for scene,directory in ((legacy,out/f'legacy_{variant}'),(cand,out/f'grammar_{variant}')):
            if private:
                for i,(_,repeat,_,_) in ext.SLOTS.items():
                    mask=np.isin(scene['meta'][:,0,0],[0,1,6,7])&(scene['meta'][:,0,1]==i);scene['meta'][mask,0,2]=repeat
                p=directory/'gpu_metadata.npz'
                with np.load(p) as old: data={k:old[k] for k in old.files}
                data['metadata']=scene['meta'];np.savez_compressed(p,**data)
        # One renderer/texture bank per tile; switch only actual mesh and metadata.
        vn=gpu.vertex_data(cand['near']);renderer=gpu.GL(vn,cand['meta'],cand['ops'],arrays,photoreal=True,cutout_mips=mips,natural=True,compact_materials=True)
        renderer.tex2(cand['proxy'],9,'reflectionProxies',False)
        if private:renderer.uniform('externalPbrLayers',ext.AUTHORED_MASK)
        if quant is None:
            quant=quant_probe(renderer,arrays)
            if efficient:
                from quality_probe32 import probe
                quant['quality32']=probe(renderer)
        def draw(verts,center,ppm,view,mode,res):
            renderer.replace_vertices(verts);p=verts[:,:3];renderer.shadow_center=(p.min(0)+p.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(p,axis=0))*1.2))
            im,metrics=renderer.render(center,ppm,view,mode,res,info);return im,renderer.last_mask.copy(),metrics
        res=(1280,960) if volume else (960,720);data=[]
        for scene in (legacy,cand):
            renderer.tex2(scene['meta'],0,'metadata',False);renderer.tex2(scene['ops'],1,'openings',False);renderer.tex2(scene['proxy'],9,'reflectionProxies',False)
            if private:renderer.uniform('externalPbrLayers',ext.AUTHORED_MASK)
            verts=gpu.vertex_data(scene['near']);lite=gpu.vertex_data(scene['lite']);rows={}
            for s in specs:
                bid=s['footprint'].building_id;camera=(-1.,1.6,.8) if s['party'] else (1.,-1.6,.8);poly=np.asarray(s['footprint'].exterior);off=np.asarray(s['offset']);lo=np.r_[poly.min(0),0]+off;hi=np.r_[poly.max(0),s['top']]+off
                center=(lo+hi)/2;pad=2.5 if volume else 1.;box=np.array([[x,y,z] for x in (lo[0]-pad,hi[0]+pad) for y in (lo[1]-pad,hi[1]+pad) for z in (lo[2],hi[2]+(3. if volume else 2.))])
                d=gpu.norm(camera);right=gpu.norm(np.cross([0,0,1],d));up=np.cross(d,right);ppm=res[1]*.83/max(np.ptp(box@up),np.ptp(box@right)*res[1]/res[0])
                a,mask,m=draw(verts[scene['no']==bid],center,ppm,camera,'day',res);a.save(out/(s['id']+('_legacy.png' if scene is legacy else '_grammar.png')))
                if efficient and s['id'] in ('walkup_sections_0','section_slab_0','tower_stacks_0','pitched_residential_0','tall_bay_tower_1'):
                    tag='31' if scene is legacy else '32';w=max((w for w in next(r for r in scene['records'] if r['id']==s['id'])['architecture']['facades'] if not w['party'] and not w['courtyard']),key=lambda w:w['length'])
                    T=(np.asarray(w['end'])-w['start'])/w['length'];N=np.asarray(w['outward'])
                    target=np.r_[np.asarray(w['start'])+T*w['length']*.38,min(s['eave']*.55,9.)]+off
                    view=(*list(N+.28*T),.14)
                    im,dm,_=draw(verts[scene['no']==bid],target,90.,view,'day',res);im.save(out/f"{s['id']}_detail_{tag}.png")
                    im,nightmask,_=draw(verts[scene['no']==bid],target,90.,view,'night',res);im.save(out/f"{s['id']}_night_{tag}.png");np.testing.assert_array_equal(dm,nightmask)
                if scene is cand:
                    b,bmask,mm=draw(lite[scene['lo']==bid],center,ppm,camera,'day',res);b.save(out/(s['id']+'_lite.png'));iou=float(np.count_nonzero(mask&bmask)/np.count_nonzero(mask|bmask));assert iou>=.94,(s['id'],iou)
                    views.append(dict(id=s['id'],near_lite_iou=iou,near_foreground=int(mask.sum()),lite_foreground=int(bmask.sum())))
                    rows[s['id']]=(a,b)
                else:rows[s['id']]=(a,)
            data.append(rows)
        for s in specs:
            key=s['id'];a=data[0][key][0];b,c=data[1][key];old=next(r for r in legacy['report']['buildings'] if r['id']==key);new=next(r for r in cand['report']['buildings'] if r['id']==key)
            diff=int(np.count_nonzero(np.any(np.asarray(a)!=np.asarray(b),axis=2)));assert diff>0
            allrows.append(dict(id=key,legacy=old,grammar=new,changed_pixels=diff));images.append((key,a,b,c,old['near'],new['near'],new['lite']))
        del renderer
    for part in range(2):
        rows=images[part*8:(part+1)*8];sheet=Image.new('RGB',(1260,len(rows)*286),(236,238,240));d=ImageDraw.Draw(sheet)
        for i,(key,a,b,c,n,m,l) in enumerate(rows):
            for j,(im,label,count) in enumerate(((a,'Stage 31' if efficient else 'Stage 30' if volume else 'legacy grammar',n),(b,'Stage 32' if efficient else 'Stage 31' if volume else 'reference grammar',m),(c,'Stage 32 Lite' if efficient else 'Stage 31 Lite' if volume else 'reference Lite',l))):
                im=im.copy();im.thumbnail((416,256));sheet.paste(im,(j*420+(420-im.width)//2,i*286));d.text((j*420+5,i*286+258),f'{key} | {label} | {count} tris',fill='black')
        sheet.save(out/f'comparison_{part}.png')
    report=dict(schema='pcg.research.reference-grammar.v1',source_kind='private_pbr' if private else 'original_shared_pbr',baseline_stage28_hashes=len(manifest),rules=len(RULES),buildings=len(allrows),cases=allrows,views=views,quantization=quant,near_max=max(r['grammar']['near'] for r in allrows),near_total=sum(r['grammar']['near'] for r in allrows),lite_total=sum(r['grammar']['lite'] for r in allrows),legacy_near_total=sum(r['legacy']['near'] for r in allrows),legacy_lite_total=sum(r['legacy']['lite'] for r in allrows),secondary_entries=sum(r['grammar']['secondary_entries'] for r in allrows),projected_columns=sum(r['grammar']['projections'] for r in allrows),roof_cores=sum(r['grammar']['roof_cores'] for r in allrows),limitations=['Authored generic rules from observed reference patterns; synthetic metric footprints, not exact donor geometry or verified building-series classification.','Same stage-26 detail planner and stage-29 material renderer used on both sides; no stage-28/29 equipment upgrade on these new fixtures.','Secondary entries and projected-front aliases require the research sidecar, not yet native ABI.','Projection fronts have shader openings, not full room/structural geometry.','No private textures in GitHub or native SDK/hardware benchmark.'])
    assert report['buildings']==16 and report['rules']==8 and report['near_max']<=1800 and report['secondary_entries']>0 and report['projected_columns']>0 and report['roof_cores']>0
    if volume:
        report['schema']='pcg.research.volume-grammar.v1';report['comparison']='stage30_vs_stage31';report['render_resolution']=[1280,960];report['refinements']=upgrades
        report['limitations']=[x for x in report['limitations'] if 'no stage-28/29 equipment upgrade' not in x]
        report['limitations'].append('Stage 28 and 29 actual geometry passes run on candidate only; the comparison includes their effects.')
        report['limitations'].append('Authoritative GIS building parts and height stepbacks are not inferred. Outward volume attachments and roof equipment are optional; no canonical shell edit.')
    if efficient:
        report['schema']='pcg.research.efficient-grammar.v1';report['comparison']='stage31_vs_stage32'
        report['limitations']=[x for x in report['limitations'] if 'candidate only' not in x]
        report['limitations'].append('Analytic sills and slab profiles are shading, not new silhouettes or self-shadowing geometry. No measured native GPU cost.')
        report['compact_saved']=sum(r['grammar']['compact_saved'] for r in allrows)
        report['savings_fraction']=1-report['near_total']/report['legacy_near_total']
        assert report['savings_fraction']>=.20,report['savings_fraction']
        assert all(r['grammar']['near']<=1800 for r in allrows)
        assert all(x['stage28'].get('sill',0)==0 and x['stage28'].get('balcony_edge',0)==0 for x in upgrades)
    dump(out/'report.json',report);dump(out/'bundle.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='bundle.json'})
    print('STAGE32' if efficient else 'STAGE31' if volume else 'STAGE30',json.dumps({k:v for k,v in report.items() if k not in ('cases','views')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('decoder',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--private-overlay',type=Path);p.add_argument('--volume',action='store_true');p.add_argument('--efficient',action='store_true');a=p.parse_args();run(a.out,a.decoder,a.private_overlay,a.volume,a.efficient)
