"""Bounded, attachment-only geometry refinement on actual stage-27 meshes.
All coordinates are tile-local meters. Never modify a source opening or slab.
Reuse the validated stage-23 metric prism and stage-27 semantic slot records.
"""
from collections import defaultdict, Counter
import hashlib
import numpy as np
from shapely.geometry import Polygon, MultiPoint
from zero_facade_trim_geometry import prism, local_coords

EPS=3e-5
CAP=1800
EXTRA_CAP=9000


def stable(*parts):return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')


def frame(w,offset):
    t=(np.asarray(w['end'])-w['start'])/w['length'];n=np.asarray(w['outward']);o=np.r_[w['start'],0.]+offset
    assert abs(np.linalg.norm(t)-1)<1e-8 and np.allclose(n,[t[1],-t[0]])
    return o,np.r_[t,0.],np.r_[n,0.]


def rectover(a,b,clear=0.):
    return min(a[2],b[2]+clear)-max(a[0],b[0]-clear)>EPS and min(a[3],b[3]+clear)-max(a[1],b[1]-clear)>EPS


def footprint(tris):
    p=tris.reshape(-1,3);return MultiPoint(p[:,:2]).convex_hull,float(p[:,2].min()),float(p[:,2].max())


def conflict(a,b):
    p,z,Z=a;q,l,H=b
    return min(Z,H)-max(z,l)>EPS and p.intersection(q).area>EPS


def plan(records,placements,scene,ops,meta,profiles,prior):
    v,u,ids,table=scene;walls={r['facade_id']:r for r in table if r.get('role')=='wall'}
    # Actual source geometry, including the projected outline of each module.
    raw=defaultdict(list);by=defaultdict(list);obs=defaultdict(list)
    for r in table:
        if 'placement' in r:
            p=v[ids==r['id']]
            if len(p):raw[r['placement']].append(p)
    for m in placements:
        bid=m['building'];key=m['id'];by[bid].append(m);lo,hi=np.asarray(m['bounds_world_local'])
        if key in raw:p=MultiPoint(np.concatenate(raw[key])[:,:2]).convex_hull
        else:p=Polygon([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]).convex_hull
        if p.area>EPS:obs[bid].append((key,(p,lo[2],hi[2])))
    old={x['building']:x for x in prior['buildings_detail']};result=[];rejected=Counter();checks=0;added=0;stats={}
    # Reserve entrance and complete balcony-column capacity for every building
    # before distributing optional sill budget. No input-order starvation.
    caps={r['footprint']['building_id']:min(CAP,r['decision']['max_triangles'])-old[r['footprint']['building_id']]['near'] for r in records}
    budgets={bid:min(cap,16+16*sum(m['kind'] in ('balcony_open','balcony_glazed') for m in by[bid])) for bid,cap in caps.items()}
    assert sum(budgets.values())<=EXTRA_CAP
    eligible=sorted((b for b in caps if profiles[b]['archetype'] not in ('glass_grid','modern_ribbon','modern_gallery')),key=lambda b:stable('budget28',b))
    while sum(budgets.values())+12<=EXTRA_CAP:
        slots=[b for b in eligible if budgets[b]+12<=caps[b]]
        if not slots:break
        for b in slots:
            if sum(budgets.values())+12>EXTRA_CAP:break
            budgets[b]+=12
    for rec in sorted(records,key=lambda q:q['footprint']['building_id']):
        f=rec['footprint'];bid=f['building_id'];off=np.asarray(rec['offset']);poly=Polygon(np.asarray(f['exterior'])+off[:2],[np.asarray(h)+off[:2] for h in f['holes']]);p=profiles[bid]
        W={w['id']:w for w in rec['architecture']['facades']};F={k:frame(w,off) for k,w in W.items()};R={};O={}
        for k,w in W.items():
            r=walls[k];start,bays,floors,_=map(int,meta[r['id'],1]);rows=ops[start:start+bays*floors+1];rects=[]
            for i,row in enumerate(rows):
                if row[2,0]==0:continue
                a=row[0].copy()
                if row[2,1]>0:a=np.r_[np.minimum(a[:2],row[1,:2]),np.maximum(a[2:],row[1,2:])]
                rects.append((i,a))
            R[k]=(r,rows,bays,floors);O[k]=rects
        budget=budgets[bid];n=0;nl=0;kept=Counter()
        def candidate(k,kind,variant,a,b,profile,family,primary,anchor,skip_slot=None):
            nonlocal checks
            w=W[k];L=w['length'];z=min(x[1] for x in profile);Z=max(x[1] for x in profile)
            if w['party'] or w['courtyard'] or not (.055<=a<b<=L-.055 and z>=0 and Z<=rec['decision']['eave_m']+EPS):return None,'facade_bounds'
            for i,box in O[k]:
                checks+=1
                if rectover((a,z,b,Z),box,0. if i==skip_slot else .006):return None,'aperture'
            t=prism(a,b,L,profile,F[k],False);en=footprint(t)
            assert en[0].intersection(poly).area<EPS,(bid,kind,'host')
            for key,other in obs[bid]:
                checks+=1
                if conflict(en,other):return None,'attachment_clearance'
            m=dict(id=anchor+':stage28:'+kind,kind=kind,variant=variant,building=bid,facade=k,anchor=anchor,primary=primary,geometry=[('stage28_'+kind,family,t)],envelope=en)
            return m,None
        def batch(seq):
            nonlocal n,nl,added,checks
            if not seq:return False
            if any(m is None for m,_ in seq):
                rejected.update(reason for m,reason in seq if m is None);return False
            ms=[m for m,_ in seq];count=sum(len(m['geometry'][0][2]) for m in ms)
            if n+count>budget or added+count>EXTRA_CAP:rejected['complete_group_budget']+=len(ms);return False
            for i,m in enumerate(ms):
                for q in ms[:i]:
                    checks+=1
                    if conflict(m['envelope'],q['envelope']):rejected['group_clearance']+=len(ms);return False
            for m in ms:
                count0=len(m['geometry'][0][2]);n+=count0;added+=count0;nl+=count0*int(m['primary']);kept[m['kind']]+=1
                obs[bid].append((m['id'],m.pop('envelope')));result.append(m)
            return True
        # All entries first. Sit on actual existing canopy top, never lift it.
        for m in by[bid]:
            if m['kind']!='canopy':continue
            k=bid+':facade:'+m['id'].split(':unified:')[1].rsplit(':',1)[0];q=local_coords(np.concatenate(raw[m['id']]),F[k]);lo=q.min(0);hi=q.max(0);a,b=lo[0]+.025,hi[0]-.025;D=hi[1];Z=hi[2]
            if p['use']==0:
                variant='sloped_hood';prof=[(.025,Z),(D-.015,Z),(D-.015,Z+.045),(.025,Z+.27)];mat='metal_roof_red'
            elif p['use']==2:
                variant='metal_fascia';prof=[(D-.14,Z),(D-.015,Z),(D-.015,Z+.12),(D-.14,Z+.12)];mat='painted_rail'
            else:
                variant='drip_coping';prof=[(.045,Z),(D-.015,Z),(D-.015,Z+.042),(D-.10,Z+.085),(.045,Z+.085)];mat='painted_trim'
            m0,why=candidate(k,'entry_cap',variant,a,b,prof,mat,True,m['id'])
            if m0 is None:
                m0,why=candidate(k,'entry_cap','low_coping',a,b,[(D-.10,Z),(D-.02,Z),(D-.02,Z+.035),(D-.10,Z+.035)],'painted_trim',True,m['id'])
            batch([(m0,why)])
        # Whole balcony columns; a partial budget never changes one slab in a stack.
        stacks=defaultdict(list)
        for m in by[bid]:
            if m['kind'] not in ('balcony_open','balcony_glazed'):continue
            k,tail=m['anchor'].rsplit(':window:',1);floor,bay=map(int,tail.split(':'));stacks[k,bay].append(m)
        for (k,bay),ms in sorted(stacks.items(),key=lambda x:stable('apron',bid,*x[0])):
            seq=[]
            for m in ms:
                slab=[v[ids==r['id']] for r in table if r.get('placement')==m['id'] and r.get('accessory')=='slab'];assert slab
                q=local_coords(np.concatenate(slab),F[k]);lo=q.min(0);hi=q.max(0);D=hi[1];z=lo[2];Z=hi[2]
                if p['archetype'] in ('central_townhouse','brick_walkup','perimeter_block','mediterranean_balcony'):
                    variant='molded_apron';prof=[(D,z-.035),(D+.035,z-.035),(D+.065,z+.018),(D+.065,Z-.03),(D,Z-.03)];family='painted_trim'
                elif p['archetype'] in ('modern_gallery','caucasus_tower'):
                    variant='metal_drip';prof=[(D,z+.025),(D+.045,z+.025),(D+.045,Z-.025),(D,Z-.025)];family='painted_rail'
                else:
                    variant='beveled_apron';prof=[(D,z),(D+.045,z+.025),(D+.045,Z-.035),(D,Z-.035)];family='concrete_panel'
                seq.append(candidate(k,'balcony_edge',variant,lo[0]+.025,hi[0]-.025,prof,family,True,m['id']))
            batch(seq)
        # Sills are near-only, grouped by whole bay. No per-window geometry
        # on office curtain walls, paired balcony blocks, party/courtyard walls.
        if p['archetype'] not in ('glass_grid','modern_ribbon','modern_gallery'):
            cols=[]
            for k,(r,rows,bays,floors) in R.items():
                if W[k]['party'] or W[k]['courtyard']:continue
                for b in range(bays):cols.append((k,b))
            for k,b in sorted(cols,key=lambda x:stable('sill',bid,*x)):
                r,rows,bays,floors=R[k];seq=[]
                for floor in range(floors):
                    slot=floor*bays+b;row=rows[slot]
                    if row[2,0]!=1 or row[2,1]!=0:continue
                    a,z,B,Z=map(float,row[0]);dep=.145 if p['use']==0 else .12;th=.065 if p['archetype'] in ('brick_walkup','central_townhouse','masonry_office','civic_block') else .045
                    prof=[(0,z-th),(dep,z-th),(dep,z-.019),(0,z-.004)]
                    seq.append(candidate(k,'sill','stone_sill' if th>.05 else 'thin_sill',a-.045,B+.045,prof,'stone_plinth' if th>.05 else 'painted_trim',False,k+':slot:'+str(slot),slot))
                if seq:batch(seq)
        stats[bid]=dict(near=old[bid]['near']+n,lite=old[bid]['lite']+nl,added=n,lite_added=nl,counts=dict(kept))
        assert stats[bid]['near']<=CAP
    assert len({m['id'] for m in result})==len(result) and 0<added<=EXTRA_CAP
    for m in result:
        for _,_,t in m['geometry']:
            q=t.astype(np.float32).astype(float);assert np.all(np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)>1e-9)
    return result,stats,dict(checks=checks,rejections=dict(rejected),added=added)
