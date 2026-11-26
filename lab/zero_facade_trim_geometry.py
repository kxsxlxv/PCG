"""Metric facade trim planner for rectangular research fixtures."""
from collections import defaultdict
import numpy as np
import trimesh
from shapely.geometry import MultiPoint

EPS=2e-6
CLEAR=.025
PROFILES={
    'panel_block':(.62,((1,.035),),.095,'concrete_panel'),
    'brick_stack':(.68,((1,.095),(3,.065)),.14,'painted_trim'),
    'mixed_use':(.54,((1,.12),(4,.055)),.12,'painted_trim'),
}

def wall_frame(v,uv,ids,r):
    q=ids==r['id']; p=v[q].astype(float); s=uv[q].astype(float)
    a=np.linalg.lstsq(np.c_[s,np.ones(len(s))],p,rcond=None)[0]
    t=np.round(a[0],8); origin=np.round(a[2],8)
    np.testing.assert_allclose(a[1],[0,0,1],atol=EPS)
    assert abs(np.linalg.norm(t)-1)<EPS and abs(t[2])<EPS
    np.testing.assert_allclose(np.c_[s,np.ones(len(s))]@a,p,atol=EPS)
    n=np.array([t[1],-t[0],0.]); return origin,t,n

def local_coords(p,frame):
    o,t,n=frame; a=np.asarray(p)-o
    return np.stack((a@t,a@n,a[...,2]),axis=-1)

def world_coords(p,frame):
    o,t,n=frame; p=np.asarray(p)
    return o+p[...,0,None]*t+p[...,1,None]*n+p[...,2,None]*np.array([0,0,1])

def blockers(scene):
    v,_,ids,table=scene; parts=defaultdict(list)
    for r in table:
        k=r.get('accessory','')
        if not k:continue
        p=v[ids==r['id']]
        if not len(p):continue
        if k.startswith('cutout_'):
            for i,t in enumerate(p.reshape(-1,3,3)):parts[(None,k+str(i))].append(t)
        else:parts[(r['building'],r['surface'].rsplit(':face:',1)[0])].append(p)
    return [(b,k,np.concatenate(pp).reshape(-1,3)) for (b,k),pp in parts.items()]

def wall_blocks(parts,r,frame):
    out=[]
    for b,k,p in parts:
        if b is not None and b!=r['building']:continue
        v=local_coords(p,frame); lo=v.min(0); hi=v.max(0)
        if hi[1]<-EPS or lo[0]>r['length']+1 or hi[0]<-1:continue
        out.append((k,lo,hi))
    return out

def intervals(length,profile,openings,blocks,miter):
    d=max(p[0] for p in profile); z0=min(p[1] for p in profile); z1=max(p[1] for p in profile)
    forbidden=[]
    for _,(a,b,c,e) in openings.values():
        if z1>b-CLEAR and z0<e+CLEAR:forbidden.append((a-CLEAR,c+CLEAR,'aperture'))
    for k,lo,hi in blocks:
        if hi[1]<-EPS or lo[1]>d+CLEAR or z1<=lo[2]+EPS or z0>=hi[2]-EPS:continue
        forbidden.append((lo[0]-CLEAR,hi[0]+CLEAR,k))
    spans=[(0.,length)]; cuts=[]
    for a,b,k in sorted(forbidden):
        # Exterior miters reach beyond canonical facade endpoints.
        if miter and a<0 and b>=-d:a,b=0.,max(.08,b)
        if miter and b>length and a<=length+d:a,b=min(length-.08,a),length
        if b<=0 or a>=length:continue
        a=max(0.,a);b=min(length,b)
        if b<=a:continue
        cuts.append((float(a),float(b),k))
        spans=[s for x,y in spans for s in ((x,min(y,a)),(max(x,b),y)) if s[1]-s[0]>.06]
    return spans,cuts

def prism(a,b,L,profile,frame,miter):
    p=np.asarray(profile,float); N=len(p)
    left=a-p[:,0] if miter and a<EPS else np.full(N,a)
    right=b+p[:,0] if miter and L-b<EPS else np.full(N,b)
    v=world_coords(np.concatenate((np.c_[left,p],np.c_[right,p])),frame)
    f=[]
    for j in range(1,N-1):f.extend([(0,j+1,j),(N,N+j,N+j+1)])
    for j in range(N):
        k=(j+1)%N;f.extend([(j,k,k+N),(j,k+N,j+N)])
    m=trimesh.Trimesh(vertices=v,faces=np.asarray(f)[:,::-1],process=False)
    if m.volume<0:m.invert()
    assert m.is_watertight and m.is_winding_consistent and m.volume>1e-8
    assert np.min(m.area_faces)>1e-9
    return m.vertices[m.faces]

def profiles(r,style):
    plinth,belts,depth,mat=PROFILES[style]; H=r['height']; fh=r['floor_h']
    yield 'plinth_drip','stone_plinth',[(0,plinth-.085),(.09,plinth-.085),(.09,plinth),(0,plinth+.035)],True
    for floor,d in belts:
        if floor>=r['floors']:continue
        z=floor*fh-.32
        yield 'belt',mat,[(0,z),(d*.50,z),(d,z+.045),(d,z+.11),(0,z+.15)],True
    # Brick already has a continuous cornice and slope soffit; another
    # protruding strip conflicts with the roof and top-floor enclosures.
    if style!='brick_stack':
        z=H-.41; top=H-.185
        yield 'cornice_under',mat,[(0,z),(depth*.42,z),(depth,z+.055),(depth,top),(0,top)],True

def plan(scene,rectangles,pipe_fn):
    v,uv,ids,table=scene;parts=blockers(scene); modules=[]; audit=[]; drains=defaultdict(list); frames={}
    walls=[r for r in table if r.get('role')=='wall' and 'accessory' not in r and not r.get('party')]
    for r in walls:frames[r['id']]=wall_frame(v,uv,ids,r)
    # Opposite pitched eave outlet. Flat roofs retain existing design.
    for r in walls:
        if r['building']!='facade:brick_stack' or r['length']<20:continue
        frame=frames[r['id']]; B=wall_blocks(parts,r,frame)
        existing=[(lo,hi) for k,lo,hi in B if 'drain' in k and hi[1]>.02 and lo[1]<.5]
        if existing:
            drains[r['id']].extend(float((lo[0]+hi[0])/2) for lo,hi in existing);continue
        H=r['height']; L=r['length']; gutter=[(lo,hi) for k,lo,hi in B if 'gutters' in k and abs(lo[2]-(H-.26))<.003 and lo[1]<.40<hi[1]]
        if not gutter:continue
        for u in (.36,L-.36):
            path=[(u,.22,.12),(u,.085,.29),(u,.085,H-.78),(u,.40,H-.39),(u,.40,H-.26)]
            p=pipe_fn(path,radius=.045,n=6); lo=p.reshape(-1,3).min(0);hi=p.reshape(-1,3).max(0)
            if any(min(hi[0],c+CLEAR)>max(lo[0],a-CLEAR) and min(hi[2],e+CLEAR)>max(lo[2],b-CLEAR) for _,(a,b,c,e) in rectangles(r).values()):continue
            if any(np.all(np.minimum(hi,h)-np.maximum(lo,l)>EPS) for _,l,h in B):continue
            if not any(l[0]<=u-.045 and h[0]>=u+.045 for l,h in gutter):continue
            p=world_coords(p,frame)[:,::-1,:]; mid=r['facade_id']+':stage23:drain'
            modules.append(dict(id=mid,kind='stage23_drain',building=r['building'],facade=r['facade_id'],geometry=[('stage23_drain','painted_rail',p)],centerline=world_coords(path,frame).tolist()))
            parts.append((r['building'],mid,p.reshape(-1,3))); drains[r['id']].append(u);break
    for r in walls:
        style=r['building'].split(':')[-1];frame=frames[r['id']];L=r['length'];B=wall_blocks(parts,r,frame);ops=rectangles(r)
        for kind,mat,profile,miter in profiles(r,style):
            spans,cuts=intervals(L,profile,ops,B,miter)
            for i,(a,b) in enumerate(spans):
                p=prism(a,b,L,profile,frame,miter);mid=r['facade_id']+':stage23:'+kind+':'+str(profile[0][1])+':'+str(i)
                q=local_coords(p.reshape(-1,3),frame);lo=q.min(0);hi=q.max(0)
                for _,(x,y,X,Y) in ops.values():
                    assert min(hi[0],X)-max(lo[0],x)<=EPS or min(hi[2],Y)-max(lo[2],y)<=EPS,(mid,'opening')
                for k,l,h in B:
                    assert not (min(hi[0],h[0])-max(lo[0],l[0])>EPS and min(hi[2],h[2])-max(lo[2],l[2])>EPS and h[1]>=-EPS and l[1]<hi[1]-EPS),(mid,k)
                modules.append(dict(id=mid,kind=kind,building=r['building'],facade=r['facade_id'],geometry=[('stage23_'+kind,mat,p)],span=[a,b],profile=profile))
            audit.append(dict(facade=r['facade_id'],kind=kind,segments=len(spans),coverage_m=sum(b-a for a,b in spans),length_m=L,cuts=cuts))
    trim=[m for m in modules if m['kind']!='stage23_drain']; checked=0
    solids=[]
    for m in trim:
        p=m['geometry'][0][2].reshape(-1,3);solids.append((m,MultiPoint(p[:,:2]).convex_hull,p[:,2].min(),p[:,2].max()))
    for i,(m,p,z,Z) in enumerate(solids):
        for n,q,y,Y in solids[:i]:
            if m['building']!=n['building'] or min(Z,Y)-max(z,y)<=EPS:continue
            assert p.intersection(q).area<1e-7,(m['id'],n['id']);checked+=1
    assert len({m['id'] for m in modules})==len(modules)
    return modules,audit,dict(drains),checked
