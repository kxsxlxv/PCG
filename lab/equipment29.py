"""Original metric equipment variants inside previously accepted envelopes.
Reuses stage-28 anchors, record IDs, material/ABI tables and cutout geometry.
No donor geometry or texture is imported. Optional equipment gains Lite proxies.
"""
from collections import Counter, defaultdict
import hashlib, json, shutil
from pathlib import Path
import numpy as np
import compile_zero_geometry_tile as tile
from material_lab import scene_io

EPS=3e-4
CAP=1800

def stable(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'little')

def groups(path):
    result={}
    for p,a,f,_ in scene_io.glb(path):
        uv=a['TEXCOORD_0'].copy();uv[:,1]=1-uv[:,1]
        result[p['extras']['pcg_role']]=dict(v=scene_io.z_up(a['POSITION']).astype(np.float32),n=scene_io.z_up(a['NORMAL']).astype(np.float32),uv0=uv,uv1=a['TEXCOORD_1'].copy(),color=a['COLOR_0'].copy(),f=f.reshape(-1,3).astype(np.uint32))
    return result

def rid(g):
    c=g['color'].astype(int);r=c[:,0]+256*c[:,1]
    assert np.all(r[g['f']]==r[g['f']][:,:1]);return r[g['f'][:,0]]

def normal(t):
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);l=np.linalg.norm(n,axis=1)
    assert np.all(l>1e-10);return n/l[:,None]

def box(lo,hi):
    # Six consistently outward-facing quads, in right-handed local axes.
    a,b,c=lo;d,e,f=hi
    v=np.array([[a,b,c],[d,b,c],[d,e,c],[a,e,c],[a,b,f],[d,b,f],[d,e,f],[a,e,f]])
    q=np.array([[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
    return v[q[:,[0,1,2,0,2,3]].reshape(-1,3)]

def pipe(x,y,z0,z1,r,sides=8):
    # Open tube with actual wall thickness and an annular top, no false cap.
    a=np.arange(sides)*2*np.pi/sides
    xy=np.stack((np.cos(a),np.sin(a)),axis=1)
    def ring(radius,z):return np.c_[xy*radius+[x,y],np.full(sides,z)]
    L,H=ring(r,z0),ring(r,z1);l,h=ring(r*.68,z0),ring(r*.68,z1)
    out=[]
    for i in range(sides):
        j=(i+1)%sides
        for q in ([L[i],L[j],H[j],H[i]],[l[j],l[i],h[i],h[j]],[H[i],H[j],h[j],h[i]]):
            out.extend((np.array(q)[[0,1,2]],np.array(q)[[0,2,3]]))
    return np.asarray(out)

def recipe(kind,variant,extent,lite=False):
    # Fractions fit the accepted envelope, not donor vertices.
    w,d,h=extent
    if kind=='ac':
        z=.20 if variant==0 else .25
        parts=[('ac','ac_body',box([0,.16*d,z*h],[w,d,h]))]
        for x in (.12,.78):
            if lite:parts.append(('ac','ac_support',box([x*w,0,0],[(x+.10)*w,.88*d,z*h])))
            else:
                parts.append(('ac','ac_support',box([x*w,0,0],[(x+.10)*w,.16*d,(z+.24)*h])))
                parts.append(('ac','ac_support',box([x*w,.10*d,.09*h],[(x+.10)*w,.88*d,z*h])))
        return parts
    # Mounts fit the accepted envelope. Tube ends meet the solid base.
    base=('vent','metal',box([.065*w,.065*d,0],[.935*w,.935*d,.17*h]))
    if variant==0:
        parts=[('vent','grille',box([.065*w,.065*d,0],[.935*w,.935*d,.76*h]))] if lite else [base,('vent','grille',box([.10*w,.10*d,.17*h],[.90*w,.90*d,.76*h]))]
        parts.append(('vent','metal',box([0,0,.76*h],[w,d,.85*h])))
        fan=.28*min(w,d)
        for x in (.32,.68):
            parts.append(('vent','fan',box([x*w-fan/2,.5*d-fan/2,.85*h],[x*w+fan/2,.5*d+fan/2,h])))
    else:
        parts=[base];r=.19*min(w,d)
        for x,z in ((.29,1.),(.71,.80)):
            parts.append(('vent','metal',pipe(x*w,.5*d,.17*h,z*h,r,6 if lite else 8)))
    return parts

def frame(record,m,tri):
    center=tri.reshape(-1,3).mean(0);off=np.asarray(record['offset'])
    walls=record['architecture']['facades']
    if m['kind']=='ac':
        def dist(w):
            p=center[:2]-off[:2]-w['start'];return abs(p@w['outward'])
        w=min(walls,key=dist);n=np.r_[w['outward'],0.];t=np.array([n[1],-n[0],0.])
    else:
        w=max(walls,key=lambda w:w['length']);t=np.r_[(np.asarray(w['end'])-w['start'])/w['length'],0.];n=np.array([-t[1],t[0],0.])
    basis=np.stack((t,n,[0,0,1.]),axis=1);np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-7);assert np.linalg.det(basis)>.99999
    q=tri.reshape(-1,3)@basis;lo=q.min(0);hi=q.max(0)
    if m['kind']=='ac':assert abs(lo[1]-np.r_[np.asarray(w['start'])+off[:2],0.]@n)<EPS
    return basis,lo,hi

def select(g,mask):
    f=g['f'][mask];used=np.unique(f);remap=np.full(len(g['v']),-1,int);remap[used]=np.arange(len(used))
    return {k:(remap[f].astype(v.dtype) if k=='f' else v[used].copy()) for k,v in g.items()}

def append(g,verts,normals,uv0,uv1,colors):
    start=len(g['v']);num=len(verts)
    for k,val in (('v',verts),('n',normals),('uv0',uv0),('uv1',uv1),('color',colors),('f',np.arange(start,start+num).reshape(-1,3))):
        g[k]=np.concatenate((g[k],np.asarray(val,dtype=g[k].dtype)))

def build(source,out,records,placements,table,prior,*,require_all_variants=True):
    out.mkdir(parents=True,exist_ok=True)
    near=groups(source/'tile_opaque.glb');lite=groups(source/'tile_lite_opaque.glb');g=near['roof'];original=g;ids=rid(g);tri=g['v'][g['f']].astype(float);norm=normal(tri)
    by=defaultdict(list)
    for i,r in enumerate(table):
        if r.get('placement'):by[r['placement']].append(i)
    rec={r['footprint']['building_id']:r for r in records};stats={k:dict(near=v['near'],lite=v['lite']) for k,v in prior['per_building'].items()}
    lite_original={role:{k:v.copy() for k,v in q.items()} for role,q in lite.items()}
    keep=np.ones(len(ids),bool);adds=[];lite_add=[];lite_parts=[];rows=[]
    targets=sorted((m for m in placements if m['kind'] in ('ac','vent')),key=lambda m:(m['building'],stable(m['id']),m['id']))
    for m in targets:
        bid=m['building'];allids=by[m['id']];mask=np.isin(ids,allids)
        replace_ids=[i for i in allids if table[i].get('accessory') in ('ac','vent','equipment_cap')];body=np.isin(ids,replace_ids)
        assert mask.any() and body.any() and not np.isin(rid(lite['roof']),allids).any()
        basis,lo,hi=frame(rec[bid],m,tri[body]);extent=hi-lo;assert min(extent)>.05
        variant=stable(m['id'])%2;parts=recipe(m['kind'],variant,extent);simple=recipe(m['kind'],variant,extent,True)
        newcount=sum(len(p[2]) for p in parts);oldcount=int(body.sum());delta=newcount-oldcount
        accept=stats[bid]['near']+delta<=CAP
        # Failed budget upgrades retain the same original shape at both LODs.
        retained=mask & ~body if accept else mask
        if retained.any():lite_add.append(select(g,retained))
        litecount=int(retained.sum())+(sum(len(p[2]) for p in simple) if accept else 0)
        stats[bid]['lite']+=litecount
        rows.append(dict(id=m['id'],building=bid,kind=m['kind'],variant=('bracket_low' if variant==0 else 'bracket_high') if m['kind']=='ac' else ('air_handler' if variant==0 else 'twin_stacks'),accepted=bool(accept),old_triangles=int(mask.sum()),near_triangles=int(mask.sum())+delta if accept else int(mask.sum()),lite_triangles=litecount,reason=None if accept else 'whole_building_budget',view=(basis[:,1]+.45*basis[:,0]+np.array([0.,0.,.32 if m['kind']=='ac' else .75])).tolist(),center=((lo+hi)/2@basis.T).tolist()))
        if not accept:continue
        keep[body]=False;stats[bid]['near']+=delta
        for dest,recipes in ((adds,parts),(lite_parts,simple)):
            for material,tag,part in recipes:
                q=(part+lo)@basis.T;flat=q.reshape(-1,3);n=normal(q)
                assert np.all(flat>=np.asarray(m['bounds_world_local'][0])-EPS) and np.all(flat<=np.asarray(m['bounds_world_local'][1])+EPS),m['id']
                assert np.all(part>=-EPS) and np.all(part<=extent+EPS)
                options=np.flatnonzero(body & np.isin(ids,[i for i in replace_ids if table[i]['accessory']==material]));assert len(options)>0
                # Reuse the nearest original directional record; no new ABI IDs.
                sel=options[np.argmax(n@norm[options].T,axis=1)];rids=ids[sel];uv=[];uv_base=[]
                for p,N,j in zip(q,n,rids):
                    T=np.cross([0,0,1.],N)
                    if np.linalg.norm(T)<1e-8:T=np.array([1.,0,0])-N*N[0]
                    T/=np.linalg.norm(T);B=np.cross(N,T)
                    proj=np.stack((flat@T,flat@B),axis=1);raw=np.stack((p@T,p@B),axis=1)-proj.min(0)
                    uv_base.append(raw/tile.MATERIALS[table[j]['family']].repeat_m)
                    raw=raw/np.maximum(np.ptp(proj,axis=0),1e-8)
                    if m['kind']=='ac':
                        if tag=='ac_support':raw[:,0]-=2
                        elif N@basis[:,1]>.99:raw[:,0]+=4+2*variant
                        else:raw[:,0]+=2
                        raw*=np.asarray(table[j]['span'])
                    else:
                        # Disjoint equipment-local UV tags; no facade UV changes.
                        code=1 if tag=='fan' and N[2]>.99 else 3 if tag=='grille' and abs(N[2])<.1 else 5
                        raw[:,0]=-code-raw[:,0]
                    uv.append(raw)
                colors=np.repeat(np.stack((rids%256,rids//256,np.zeros(len(rids),int),np.full(len(rids),255)),axis=1),3,axis=0)
                dest.append((flat,np.repeat(n,3,axis=0),np.concatenate(uv_base),np.concatenate(uv),colors))
    unchanged=select(g,keep);near['roof']=unchanged
    for a in adds:append(unchanged,*a)
    for a in lite_add:
        g=lite['roof'];start=len(g['v'])
        for k in g:g[k]=np.concatenate((g[k],a[k]+start if k=='f' else a[k]))
    for a in lite_parts:append(lite['roof'],*a)
    assert all(v['lite']<=v['near']<=CAP for v in stats.values())
    assert len(rows)==len(targets) and all(v['lite_triangles']<=v['near_triangles'] for v in rows)
    if require_all_variants:assert len({r['variant'] for r in rows if r['accepted']})==4
    # Preserve the previous triangle streams for all unaffected Near and Lite faces.
    for k in ('v','n','uv0','uv1','color'):
        np.testing.assert_array_equal(original[k][original['f'][keep]],near['roof'][k][near['roof']['f'][:int(keep.sum())]])
    for role,q in lite_original.items():
        for k,v in q.items():np.testing.assert_array_equal(v,lite[role][k][:len(v)])
    for name,value in (('tile_opaque.glb',near),('tile_lite_opaque.glb',lite)):
        tile.export_glb(value,out/name);tile.read_glb(out/name,value,table)
    for name in ('roof_balcony_cutout.glb','antenna_cutout.glb','facade_params.bin','facade_cells.bin','opening_blocks_v2.json','plans.json','parameters.json','cutout_records.json'):
        shutil.copyfile(source/name,out/name)
    updates={r['id']:r for r in rows}
    updated=[dict(m,triangles=updates[m['id']]['near_triangles'],lite_triangles=updates[m['id']]['lite_triangles'],equipment_variant=updates[m['id']]['variant'] if updates[m['id']]['accepted'] else 'original') if m['id'] in updates else m for m in placements]
    (out/'placements.json').write_text(json.dumps(updated,sort_keys=True,indent=2)+'\n')
    report=dict(placements=rows,per_building=stats,variants=dict(Counter(r['variant'] for r in rows if r['accepted'])),accepted=sum(r['accepted'] for r in rows),total=len(rows),near_triangles=sum(v['near'] for v in stats.values()),lite_triangles=sum(v['lite'] for v in stats.values()),max_building_triangles=max(v['near'] for v in stats.values()),retained_lite_anchors=len(rows))
    (out/'equipment_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report
