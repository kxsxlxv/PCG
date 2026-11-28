"""Measured, dimensionless architectural transfer, independent of donor units.
Both packs supply measured contours; the High-poly patch catalog supplies
roughness-derived aperture/bay proxies. No raw donor triangles are copied.
"""
from pathlib import Path
import argparse,hashlib,json,math,shutil
from collections import Counter
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import LineString,shape
from pcg.reference_grammar import MeasuredProfile
from reference_integrated33 import verify,bank,append_arrays,assignments,dump,sha

def runs(mask):
    ids=np.flatnonzero(mask)
    return [a for a in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(a)]

def contour_features(row):
    # A reliable long-lived, near-orthogonal, connected middle tier only.
    span=np.array(row['span_source_units'],float)
    if span.shape!=(3,) or not np.isfinite(span).all() or min(span)<=0:return None
    if row['orthogonal_confidence']<.72 or row['closed_slices']<8 or row['vertical_area_fraction']<.40:return None
    tiers=row['contours']
    cand=[q for q in tiers if q['z0']<.65 and q['z1']>.35 and q['z1']-q['z0']>=.18]
    if not cand:return None
    q=max(cand,key=lambda q:(q['z1']-q['z0'],q['samples']))
    p=shape(q['polygon'])
    if p.geom_type!='Polygon' or not p.is_valid or p.area<.18:return None
    low=np.array(p.bounds[:2]);high=np.array(p.bounds[2:]);size=high-low
    if min(size)<.18:return None
    # Envelope depth of a contiguous protruding band relative to median
    # side line. Require nontrivial support and reject edge/corner excursions.
    relief=[];N=96
    for axis in (0,1):
        across=1-axis
        # Use long faces; axis is the coordinate along the face.
        if size[axis]*span[axis]<.8*size[across]*span[across]:continue
        samples=np.linspace(low[axis]+.04*size[axis],high[axis]-.04*size[axis],N)
        ext=[]
        for s in samples:
            a=[0.,0.];b=[0.,0.];a[axis]=b[axis]=s;a[across]=low[across]-.01;b[across]=high[across]+.01
            cut=p.intersection(LineString([a,b]))
            ext.append((cut.bounds[across],cut.bounds[2+across]) if not cut.is_empty else (np.nan,np.nan))
        ext=np.array(ext)
        for side in (0,1):
            t=ext[:,side]*(1 if side else -1)
            if np.isfinite(t).sum()<.95*N:continue
            delta=t-np.nanmedian(t)
            for band in runs(delta>.018*size[across]):
                if band[0]<2 or band[-1]>N-3 or not 4<=len(band)<=N*.48:continue
                depth=float(np.nanmedian(delta[band])*span[across]/min(span[:2]*size))
                width=float(len(band)/N*.92);phase=float((samples[band].mean()-low[axis])/size[axis])
                if .012<=depth<=.35:relief.append(dict(depth=depth,width=width,phase=phase,axis=axis,side=side,samples=len(band)))
    relief.sort(key=lambda r:(-r['depth']*r['width'],r['axis'],r['side'],r['phase']))
    roof=None
    last=tiers[-1];cap=shape(last['polygon']);dz=last['z1']-last['z0']
    if cap.is_valid and not cap.is_empty and last['z0']>=.70 and .015<=dz<=.18:
        ratio=cap.area/p.area
        if .025<=ratio<=.40 and cap.intersection(p).area>=cap.area*.97:
            roof=dict(area_fraction=float(ratio),height_over_body=float(dz/last['z0']),tier=last['z0'])
    return dict(id=row['id'],pack=row['pack'],aspect=float(max(span[:2]*size)/min(span[:2]*size)),
        slenderness=float(span[2]/min(span[:2]*size)),body_z=[q['z0'],q['z1']],closed_slices=row['closed_slices'],
        relief=relief[:8],roof=roof)

def facade_features(catalog):
    result={}
    for key,f in sorted(catalog['families'].items()):
        # Use only the dominant roughness components, not unverified absolute units.
        modules=[m for m in catalog['modules'] if m['family']==key and m['kind']=='glazing_patch' and m.get('glass_class')=='narrow'
                 and .18<=m['glass_span']<=.80 and .18<=m['glass_height']<=.85]
        if len(modules)<2:
            modules=[m for m in catalog['modules'] if m['family']==key and m['kind']=='glazing_patch'
                     and .15<=m.get('glass_span',0)<=.90 and .15<=m.get('glass_height',0)<=.9]
        if not modules:continue
        med=lambda v:float(np.median(v))
        spacing=med([m['ratio'] for m in modules])
        width=med([m['ratio']*m['glass_span'] for m in modules])
        height=med([m['glass_height'] for m in modules])
        sill=med([max(0.,1.-m['glass_center']-.5*m['glass_height']) for m in modules])
        result[key]=dict(spacing_over_floor=float(np.clip(spacing,.4,3.5)),width_over_floor=float(np.clip(width,.05,3.5)),
                         height_over_floor=float(np.clip(height,.04,1.)),sill_over_floor=float(np.clip(sill,0,1)),
                         modules=[m['id'] for m in modules],classification='roughness-derived glazing extent; not certified window or balcony class')
    return result

def choose(spec,features,facades):
    f=spec['footprint'];poly=f.polygon();coords=list(poly.minimum_rotated_rectangle.exterior.coords)
    sides=sorted(math.dist(a,b) for a,b in zip(coords[:-1],coords[1:]) if math.dist(a,b)>1e-9)
    aspect=sides[-1]/sides[0];slender=spec['eave']/sides[0]
    def distance(x):return abs(math.log(x['aspect']/aspect))+.8*abs(math.log(x['slenderness']/slender))
    # Low-poly profiles for walkups, High-poly for taller apartment stock.
    # This is an explicit design policy, not a series identification.
    pack='low' if spec['rule']=='walkup_sections' else 'high'
    pool=sorted([x for x in features if x['pack']==pack],key=lambda x:(distance(x),x['id']))
    if not pool:raise ValueError('no eligible contour observations: '+pack)
    # Choose relief-bearing donor within modest shape-distance slack.
    near=[x for x in pool if distance(x)<=distance(pool[0])+.85]
    relief=next((x for x in near if x['relief']),None)
    geom=relief or pool[0]
    roof=next((x for x in near if x['roof'] is not None),None)
    fd=sorted((x for x in features if x['id'] in facades),key=lambda x:(distance(x),x['id']))
    # Family contour may be less reliable than its facade image; fall back
    # to the nearest available family by bounded glazing/bay ratio.
    family=fd[0]['id'] if fd else sorted(facades,key=lambda k:(abs(facades[k]['spacing_over_floor']-1),k))[0]
    fs=facades[family]
    args={k:fs[k] for k in ('spacing_over_floor','width_over_floor','height_over_floor','sill_over_floor')}
    if relief:
        a=relief['relief'][0];args.update(relief_over_short_side=a['depth'],relief_phase=a['phase'],relief_width_fraction=a['width'])
    if roof:
        a=roof['roof'];args.update(roof_area_fraction=a['area_fraction'],roof_height_over_body=a['height_over_body'])
    profile=MeasuredProfile(**args)
    return profile,dict(building=f.building_id,geometry_donor=geom['id'],geometry_pack=geom['pack'],geometry_distance=distance(geom),
        relief_donor=relief['id'] if relief else None,roof_donor=roof['id'] if roof else None,facade_family=family,
        facade_modules=fs['modules'],profile=asdict(profile),target_aspect=aspect,target_slenderness=slender)

def run(source,atlas,measured,out,decoder):
    import zero_reference_grammar as ref
    import gpu_lab as gpu
    source_files=verify(source);atlas_files=verify(atlas);measurement_files=verify(measured);verify(ref.ROOT)
    out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((measured/'report.json').read_text())['models'];features=[q for r in rows if (q:=contour_features(r)) is not None]
    facades=facade_features(json.loads((atlas/'modules.json').read_text()))
    assert len(facades)>=3 and {x['pack'] for x in features}=={'low','high'}
    dump(out/'measured_features.json',dict(contours=features,facades=facades))
    layers,cat=bank(atlas);arrays,mips,info=ref.load_materials(out,None);base=len(arrays[0]);append_arrays(arrays,layers)
    for i,maps in enumerate(layers):
        p=out/'donor_bank'/str(base+i);p.mkdir(parents=True)
        for name,a in zip(('albedo','normal','orm'),maps):Image.fromarray(a).save(p/(name+'.png'))
    dump(out/'donor_layers.json',dict(base=base,layers=cat))
    old=json.loads((source/'report.json').read_text());result=[];selected=[];sheets=[];totals=[];closeups=[]
    for variant in (0,1):
        specs=list(ref.fixtures(variant));cal={};families={}
        for spec in specs:
            if ref.RULES[spec['rule']].use!='apartments':continue
            p,q=choose(spec,features,facades)
            bid=spec['footprint'].building_id;cal[bid]=p;families[bid]=q['facade_family'];selected.append(q)
        dst=out/f'grammar_{variant}';scene=ref.compile_case(specs,'efficient',dst,decoder,calibration=cal)
        table=json.loads((dst/'parameters.json').read_text());prev_table=json.loads((source/f'grammar_{variant}'/'parameters.json').read_text())
        previous=json.loads((source/f'grammar_{variant}'/'plans.json').read_text())
        for a,b in zip(previous,scene['records']):
            assert a['footprint']==b['footprint'] and a['decision']['top_m']==b['decision']['top_m'] and a['decision']['eave_m']==b['decision']['eave_m']
        ext,stats=assignments(scene['meta'],scene['ops'],table,cat,base,families)
        rev,stats2=assignments(scene['meta'],scene['ops'],list(reversed(table)),cat,base,families);np.testing.assert_array_equal(ext,rev);assert stats==stats2
        np.savez_compressed(dst/'donor_opening_sidecar.npz',openings=ext)
        totals.append(stats)
        vn=gpu.vertex_data(scene['near']);vl=gpu.vertex_data(scene['lite'])
        renderer=gpu.GL(vn,scene['meta'],ext,arrays,photoreal=True,cutout_mips=mips,natural=True,compact_materials=True)
        renderer.tex2(scene['proxy'],9,'reflectionProxies',False)
        def draw(verts,center,ppm,view,mode,enable):
            renderer.uniform('q33Enabled',int(enable));renderer.replace_vertices(verts)
            pts=verts[:,:3];renderer.shadow_center=(pts.min(0)+pts.max(0))/2;renderer.shadow_extent=max(20.,float(np.linalg.norm(np.ptp(pts,axis=0))*1.2))
            im,_=renderer.render(center,ppm,view,mode,(1280,960),info)
            return im,renderer.last_mask.copy()
        for spec in specs:
            bid=spec['footprint'].building_id;key=spec['id'];poly=np.asarray(spec['footprint'].exterior);off=np.asarray(spec['offset'])
            a=np.r_[poly.min(0),0]+off;b=np.r_[poly.max(0),spec['top']]+off
            center=(a+b)/2;camera=(-1.,1.6,.8) if spec['party'] else (1.,-1.6,.8)
            box=np.array([[x,y,z] for x in (a[0]-2.5,b[0]+2.5) for y in (a[1]-2.5,b[1]+2.5) for z in (a[2],b[2]+3.)])
            d=gpu.norm(camera);right=gpu.norm(np.cross([0,0,1],d));up=np.cross(d,right);ppm=960*.83/max(np.ptp(box@up),np.ptp(box@right)*960/1280)
            nv,lv=vn[scene['no']==bid],vl[scene['lo']==bid]
            A=Image.open(source/(key+'_grammar.png')).convert('RGB');B,mb=draw(nv,center,ppm,camera,'day',False)
            C,mc=draw(nv,center,ppm,camera,'day',True);D,md=draw(lv,center,ppm,camera,'day',True)
            np.testing.assert_array_equal(mb,mc)
            iou=float((mc&md).sum()/max(1,(mc|md).sum()));assert iou>=.94,(key,iou)
            before=next(r['grammar'] for r in old['cases'] if r['id']==key);after=next(r for r in scene['report']['buildings'] if r['id']==key)
            assert len(nv)//3==after['near'] and len(lv)//3==after['lite'] and after['near']<=1800
            rec=next(r for r in scene['records'] if r['id']==key);prev=next(r for r in previous if r['id']==key)
            def geom(r):
                q=r['reference_grammar']
                return dict(projections=q['projections'],roof_cores=q['roof_cores'],entries=q['entries'])
            result.append(dict(id=key,calibrated=bid in cal,old_near=before['near'],near=after['near'],lite=after['lite'],old_lite=before['lite'],
                old_openings=before['openings'],openings=after['openings'],old_projections=before['projections'],projections=after['projections'],
                old_roof_cores=before['roof_cores'],roof_cores=after['roof_cores'],module_counts=after['module_counts'],near_lite_iou=iou,
                geometry_intents_changed=geom(rec)!=geom(prev),old_profile=prev['architecture']['profile'],profile=rec['architecture']['profile'],
                donor_windows=stats['buildings'].get(bid,0),changed_pixels=int(np.any(np.asarray(A)!=np.asarray(C),axis=2).sum())))
            for tag,im in zip(('32','geometry','33','33_lite'),(A,B,C,D)):im.save(out/f'{key}_{tag}.png')
            sheets.append((key,A,B,C,D))
            if key in ('walkup_sections_0','section_slab_0','tower_stacks_0','tall_bay_tower_1'):
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
                        sheet.paste(im,(j*1280,k*995));dd.text((j*1280+8,k*995+965),f'{key} | {mode} | '+('calibrated geometry' if j==0 else 'donor PBR'),fill='black')
                sheet.save(out/f'{key}_closeup.png');closeups.append(key)
        del renderer
    for part in range(4):
        subset=sheets[part*4:(part+1)*4];sheet=Image.new('RGB',(1600,len(subset)*264),(236,238,240));d=ImageDraw.Draw(sheet)
        for i,(key,*ims) in enumerate(subset):
            for j,im in enumerate(ims):
                im=im.copy();im.thumbnail((396,240));sheet.paste(im,(j*400+(400-im.width)//2,i*264))
                d.text((j*400+5,i*264+242),key+' | '+('32','geometry','33 PBR','33 Lite')[j],fill='black')
        sheet.save(out/f'comparison_{part}.png')
    assert len(result)==16 and len(selected)==10
    assert {q['geometry_pack'] for q in selected}=={'low','high'}
    assert sum(r['geometry_intents_changed'] for r in result if r['calibrated'])>=3
    report=dict(schema='pcg.research.measured-volume.v1',source_verified_files=source_files,atlas_verified_files=atlas_files,measurement_verified_files=measurement_files,
        eligible_contours=dict(Counter(x['pack'] for x in features)),calibration=selected,cases=result,assignments=totals,closeups=closeups,
        near_total=sum(r['near'] for r in result),lite_total=sum(r['lite'] for r in result),near_max=max(r['near'] for r in result),
        limitations=['Only apartment profiles calibrated. Nonresidential and pitched house geometry retain original rules.',
          'Donor-relative ratios are bounded design transfer, not metric donor measurement or semantic building series ID.',
          'Projection relief can also be facade offsets; no claim that the source relief is a balcony or stairwell.',
          'Top-tier ratios are optional roof cores; no inferred GIS height steps or detached extensions.',
          'Synthetic GIS fixture compile, actual GLB readback and software EGL; no native UNIGINE runtime or measured hardware performance.'])
    dump(out/'report.json',report);dump(out/'bundle.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='bundle.json'})
    print('CALIBRATED33',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','atlas','measured','out','decoder'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.source,a.atlas,a.measured,a.out,a.decoder)
