"""Consolidated stage 21/22 geometry. Rebuild whole projecting balcony modules.
Metric local coordinates; preserve host geometry and original cell ABI. New
opening rectangles are an explicit research sidecar, not silently PCGF v1.
"""
from collections import defaultdict, Counter
import hashlib
import numpy as np

EPS=3e-5

def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little')
def wall_frame(v,uv,ids,r):
    p=v[ids==r['id']];q=uv[ids==r['id']]
    a=np.linalg.lstsq(np.c_[q,np.ones(len(q))],p,rcond=None)[0]
    t=np.round(a[0],7);o=np.round(a[2],7);n=np.array([t[1],-t[0],0.])
    np.testing.assert_allclose(a[1],[0,0,1],atol=EPS)
    np.testing.assert_allclose(np.c_[q,np.ones(len(q))]@a,p,atol=EPS)
    return o,t,n

def local(p,F):
    o,t,n=F;q=np.asarray(p)-o
    return np.stack((q@t,q@n,q[...,2]),axis=-1)
def world(p,F):
    o,t,n=F;p=np.asarray(p)
    return o+p[...,0,None]*t+p[...,1,None]*n+p[...,2,None]*np.array([0,0,1])
def bounds(p):
    p=np.asarray(p).reshape(-1,3);return p.min(0),p.max(0)
def overlap(a,b,tol=EPS):return bool(np.all(np.minimum(a[1],b[1])-np.maximum(a[0],b[0])>tol))
def rectover(a,b,clear=0.):return min(a[2],b[2]+clear)-max(a[0],b[0]-clear)>EPS and min(a[3],b[3]+clear)-max(a[1],b[1]-clear)>EPS

def cuboid(lo,hi,F):
    assert np.all(np.asarray(hi)-lo>EPS)
    v=np.array([[x,y,z] for z in (lo[2],hi[2]) for x,y in ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))])
    f=[(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
    for i in range(4):j=(i+1)%4;f.extend([(i,j,j+4),(i,j+4,i+4)])
    return world(v,F)[np.asarray(f)[:,::-1]] # local frame is left-handed

def quad(a,b,z,Z,F):
    a=np.array([*a,z]);b=np.array([*b,z]);c=b+[0,0,Z-z];d=a+[0,0,Z-z]
    return world(np.array([a,b,c,d]),F)[[[0,1,2],[0,2,3]]]

def openings(r,cells):
    D={};n=r['floors']*r['bays'];a=r['cell_start']
    for j in np.flatnonzero(cells[a:a+n]):
        code=int(cells[a+j]);f,b=divmod(int(j),r['bays']);role=code&3;c=r['first_center']+b*r['spacing'];z=f*r['floor_h']+(r['balcony_threshold'] if role==2 else r['sill']);h=r['door_height'] if role==2 else r['window_height'];w=r['window_width']
        D[int(j)]=(code,(c-w/2,z,c+w/2,z+h))
    if r['door'] is not None:
        x,y,w,h=r['door'];D[n]=(3,(x,y,x+w,y+h))
    return D

def plan(scene,cells):
    v,uv,ids,table=scene;walls={r['facade_id']:r for r in table if r.get('role')=='wall' and 'accessory' not in r};frames={k:wall_frame(v,uv,ids,r) for k,r in walls.items()}
    old=defaultdict(list);other=defaultdict(list)
    for r in table:
        k=r.get('accessory','');p=v[ids==r['id']]
        if not k or not len(p):continue
        place=r.get('placement','')
        if place.endswith(':balcony'):old[place].append((r,p));continue
        if k=='cutout_balcony':continue
        if k.startswith('cutout_'):
            for i,t in enumerate(p.reshape(-1,3,3)):other[(None,k+str(i))].append(t)
        else:other[(r['building'],place)].append(p)
    obstacles=[(b,k,np.concatenate(pp).reshape(-1,3)) for (b,k),pp in other.items()]
    original_boxes={k:bounds(np.concatenate([p for r,p in pp])) for k,pp in old.items()}
    revised={};modules=[];guards=[];sidecar={};dead=set();checked=0
    for k,pp in sorted(old.items()):
        fid=k.split(':window:')[0];r=walls[fid];F=frames[fid];b=r['building'];f,bay=map(int,k.split(':window:')[1].split(':')[:2]);slot=f*r['bays']+bay
        op=openings(r,cells);code,original=op[slot];assert code&3==2
        c=(original[0]+original[2])/2;z=original[1];oldlo,oldhi=bounds(local(np.concatenate([p for _,p in pp]),F));dep=float(oldhi[1]);bottom=float(oldlo[2]);state='open' if {a['accessory'] for a,_ in pp}=={'slab'} else 'glazed'
        assert abs(oldlo[1])<EPS and abs(oldhi[0]-oldlo[0]-1.7)<EPS and dep>.5
        top=z+2.28 if state=='glazed' else z+.95
        obs=[(q,bounds(local(p,F))) for bb,q,p in obstacles if bb is None or bb==b]
        # Preserve clearance against not-yet-replaced balconies as well.
        for q,p in original_boxes.items():
            if q==k:continue
            if not q.startswith(b+':'):continue
            a,B=p;corners=np.array([[x,y,Z] for x in (a[0],B[0]) for y in (a[1],B[1]) for Z in (a[2],B[2])]);obs.append((q,bounds(local(corners,F))))
        for q,p in revised.items():
            if q.startswith(b+':'):obs.append((q,bounds(local(p,F))))
        selected=None
        # First use the metric 2.10m paired opening and 2.40m slab. Fall
        # back to a narrower paired block where an existing dish/AC demands it.
        for width,slab in ((2.10,2.40),(1.90,2.20),(1.65,1.95),(1.40,1.70)):
            a=c-slab/2;A=c+slab/2
            bb=(np.array([a,0,bottom]),np.array([A,dep,top]))
            if a<.08 or A>r['length']-.08:continue
            env=(c-width/2,z,c+width/2,original[3])
            if any(rectover(env,box,.04) for j,(_,box) in op.items() if j!=slot):continue
            # Existing geometry originally touches host plane; only positive
            # outward volume intersection counts. Coping/cornice may touch.
            if any(overlap(bb,q) for _,q in obs):continue
            selected=(width,slab,a,A,env);checked+=len(obs);break
        if selected is None:raise AssertionError(('no safe paired block',k,oldlo.tolist(),oldhi.tolist()))
        width,slab,a,A,env=selected
        geom=[('slab','concrete_panel',cuboid([a,0,bottom],[A,dep,z],F))]
        edges=[(a,0),(a,dep),(A,dep),(A,0)]
        if state=='glazed':
            # Uniform glazing over front and BOTH sides. No half roof or half upper face.
            for j in range(3):geom.append(('stage21_glazed','painted_rail',quad(edges[j],edges[j+1],z,top-.075,F)))
            geom.append(('stage21_cap','bitumen_roof',cuboid([a,0,top-.075],[A,dep,top],F)))
        else:
            guards.append(dict(id=k,path=world(np.c_[edges,np.full(4,z)],F).tolist(),height=.95))
        p=np.concatenate([q.reshape(-1,3) for _,_,q in geom]);revised[k]=p
        modules.append(dict(id=k,kind='balcony_'+state,building=b,facade=fid,geometry=geom,slot=slot,width_m=slab,depth_m=dep,state=state))
        # Paired asymmetric block, one common room state. Sill of window at
        # normal window level; door floor stays exactly on slab.
        x,_,X,Y=env;doorw=min(.80,width*.45);post=.085;winw=width-doorw-post;assert winw>.45
        if seed(k)%2==0:door=(x,z,x+doorw,Y);win=(x+doorw+post,f*r['floor_h']+r['sill'],X,Y)
        else:door=(X-doorw,z,X,Y);win=(x,f*r['floor_h']+r['sill'],X-doorw-post,Y)
        assert not rectover(door,win) and door[0]>=a and win[2]<=A
        sidecar.setdefault(str(r['id']),{})[str(slot)]=dict(code=code,door=door,window=win,post=(min(door[2],win[2]),win[1],max(door[0],win[0]),Y),room_id=k,slab_top=z,width_m=width)
        dead.update(a['id'] for a,_ in pp)
    # Verify paired rectangles against all neighbors and the existing midpoint
    # panel-joint grammar. Actual shader suppresses any old seam under the
    # enlarged rectangles; no new geometry is placed inside the host wall.
    for fid,r in walls.items():
        op=openings(r,cells);pairs=sidecar.get(str(r['id']),{});rects=[]
        for j,(code,box) in op.items():
            q=pairs.get(str(j));rects.extend([q['door'],q['window']] if q else [box])
        for i,a in enumerate(rects):
            assert 0<=a[0]<a[2]<=r['length'] and 0<=a[1]<a[3]<=r['height']
            for B in rects[:i]:assert not rectover(a,B)
    # Entrance geometry for all actual ground-door anchors. Retain the
    # existing canopy on mixed-use; small landing + new canopy where safe.
    entrances=[]
    for fid,r in walls.items():
        if r['door'] is None:continue
        F=frames[fid];x,z,w,h=r['door'];b=r['building'];c=x+w/2
        obs=[(q,bounds(local(p,F))) for bb,q,p in obstacles if bb is None or bb==b]
        geom=[];landing=(np.array([x-.20,0,0]),np.array([x+w+.20,.65,z]))
        if z>.015 and not any(overlap(landing,q) for _,q in obs):geom.append(('stage22_landing','stone_plinth',cuboid(*landing,F)))
        canopy=any('entry_canopy' in k for k,_ in obs)
        if not canopy:
            for zz in (z+h+.13,z+h+.22):
                cb=(np.array([c-.91,0,zz]),np.array([c+.91,.80,zz+.12]))
                if any(overlap(cb,q) for _,q in obs):continue
                if any(rectover((cb[0][0],cb[0][2],cb[1][0],cb[1][2]),B,.025) for _,B in openings(r,cells).values()):continue
                geom.append(('stage22_canopy','concrete_panel',cuboid(*cb,F)));canopy=True;break
        if geom:modules.append(dict(id=fid+':stage22_entry',kind='entrance',building=b,facade=fid,geometry=geom))
        entrances.append(dict(facade=fid,record=r['id'],canopy=canopy,landing=any(k=='stage22_landing' for k,_,_ in geom),material='glazed_lobby' if b.endswith('mixed_use') else 'metal_door'))
    assert len(old)==86 and len(guards)==31
    return modules,guards,sidecar,dead,entrances,checked
