"""Ray exit from the *whole* projecting enclosure in facade coordinates.
The room is evaluated exclusively on the real rear facade. Other exit faces
are structural slab/cap or a second glass surface, never room panoramas.
"""
from collections import defaultdict, Counter
import numpy as np
EPS=1e-5

def affine(v,uv):
    a=np.linalg.lstsq(np.c_[uv,np.ones(len(uv))],v,rcond=None)[0]
    np.testing.assert_allclose(np.c_[uv,np.ones(len(uv))]@a,v,atol=3e-5)
    return a

def fit(scene):
    v,uv,ids,table=scene;walls={r['facade_id']:r for r in table if r.get('role')=='wall' and 'accessory' not in r};groups=defaultdict(list)
    for r in table:
        if r.get('accessory')=='stage21_glazed' and np.any(ids==r['id']):groups[r['placement']].append(r)
    models={}
    for name,records in sorted(groups.items()):
        assert len(records)==3
        host=walls[name.split(':window:')[0]];mask=ids==host['id'];a=affine(v[mask],uv[mask]);t=a[0]/np.linalg.norm(a[0]);up=a[1]/np.linalg.norm(a[1]);n=np.cross(t,up);n/=np.linalg.norm(n);o=a[2]
        # Host UV axes are metric. Orient normal toward projecting enclosure.
        pts=np.concatenate([v[ids==r['id']] for r in records]);n*=1 if np.mean((pts-o)@n)>0 else -1
        F=np.column_stack([t,n,up]);q=(pts-o)@F;lo=q.min(0);hi=q.max(0)
        assert abs(lo[1])<3e-5 and hi[1]>.5 and hi[0]-lo[0]>=1.69
        # Numerical noise in glTF roundtrip, snap host plane exactly to 0.
        lo[1]=0.
        faces={}
        for r in records:
            m=ids==r['id'];B=affine(v[m],uv[m]);C=B@F;C[2]=(B[2]-o)@F
            p=(v[m]-o)@F
            face=next(i for i,s in ((0,lo[0]),(1,hi[0]),(3,hi[1])) if np.max(np.abs(p[:,i//2]-s))<3e-5)
            faces[face]=dict(record=r,affine=C)
        assert set(faces)=={0,1,3}
        models[name]=dict(host=host,origin=o,frame=F,lo=lo,hi=hi,faces=faces)
    return models

def trace(p,d,lo,hi):
    """Exit point from a closed AABB. Face 2 = rear host; 4/5 = slab/cap.
    Caller supplies a ray pointing inward through a visible entry face.
    d may be one common direction or an array of per-pixel directions.
    """
    p=np.asarray(p,float);d=np.broadcast_to(np.asarray(d,float),p.shape)
    p=np.clip(p,lo,hi);span=hi-lo
    assert np.all(span>0)
    edge=np.where(d>0,hi,lo)
    ts=np.full_like(p,np.inf)
    np.divide(edge-p,d,out=ts,where=np.abs(d)>1e-10)
    axis=np.argmin(ts,axis=1);idx=np.arange(len(p));t=ts[idx,axis]
    assert np.isfinite(t).all() and (t>=-EPS).all()
    hit=np.clip(p+t[:,None]*d,lo,hi)
    face=axis*2+(d[idx,axis]>0)
    return hit,face,t

def face_points(model,face,uv):
    a=model['faces'][face]['affine'];return np.c_[uv,np.ones(len(uv))]@a

def sample_rear(model,hit,face,shade,mode):
    """Callback evaluates the actual host wall's shader, including its
    existing paired balcony door and window masks. No independent room ID.
    """
    rgb=np.zeros((len(hit),3));em=np.zeros_like(rgb);rear=face==2
    if rear.any():rgb[rear],em[rear]=shade(hit[rear][:,[0,2]],model['host'],mode)
    return rgb,em,rear
