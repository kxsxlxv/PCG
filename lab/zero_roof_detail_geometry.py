"""Stage 24 roof geometry. All lengths are meters, in exported world space.
Uses actual reloaded roof planes and existing equipment placements, not a
second independent building generator. Rectangular flat/gable fixtures.
"""
from collections import defaultdict
import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

EPS=1e-5

def bbox(p):
    p=np.asarray(p).reshape(-1,3)
    return p.min(0),p.max(0)

def mesh(v,f):
    m=trimesh.Trimesh(vertices=np.asarray(v,float),faces=np.asarray(f,int),process=False)
    if m.volume<0:m.invert()
    assert m.is_watertight and m.is_winding_consistent and m.volume>1e-9
    assert np.min(m.area_faces)>1e-10
    return m.vertices[m.faces]

def solid_box(lo,hi):
    lo=np.asarray(lo);hi=np.asarray(hi);assert np.all(hi-lo>EPS)
    v=np.array([[x,y,z] for z in (lo[2],hi[2]) for x,y in
        ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))])
    f=[(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
    for j in range(4):
        k=(j+1)%4;f.extend([(j,k,k+4),(j,k+4,j+4)])
    return mesh(v,f)

def tube(x,y,z,r,h,n=8):
    # Closed annular wall, genuinely open throat; cap underneath is existing
    # equipment roof, not a new filled cylinder across the opening.
    p=[]
    for zz,rr in ((z,r),(z+h,r),(z+h,r*.72),(z,r*.72)):
        p.extend((x+rr*np.cos(2*np.pi*j/n),y+rr*np.sin(2*np.pi*j/n),zz) for j in range(n))
    f=[]
    for a in range(4):
        b=(a+1)%4
        for j in range(n):
            k=(j+1)%n;f.extend([(a*n+j,a*n+k,b*n+k),(a*n+j,b*n+k,b*n+j)])
    return mesh(p,f)

def parts(scene):
    v,uv,ids,table=scene;roof=defaultdict(list);equipment=defaultdict(lambda:defaultdict(dict));frames={}
    for r in table:
        if r.get("role")!="roof":continue
        p=v[ids==r["id"]].astype(float)
        if not len(p):continue
        k=r.get("accessory")
        if k is None:
            b=r["building"];roof[b].extend(p.reshape(-1,3,3))
            s=uv[ids==r["id"]].astype(float)
            a=np.linalg.lstsq(np.c_[s,np.ones(len(s))],p,rcond=None)[0]
            np.testing.assert_allclose(np.c_[s,np.ones(len(s))]@a,p,atol=2e-5)
            frames[r["id"]]=a
        else:
            d=equipment[r["building"]][r["placement"]]
            d.setdefault(k,[]).append(p)
    return {k:np.asarray(p) for k,p in roof.items()},equipment,frames

def planes(tris):
    # Use XY triangle containment and interpolated Z, including both sides
    # at a gable ridge. Avoid assuming the roof pitch from a style label.
    out=[]
    for t in tris:
        a=np.linalg.solve(np.c_[t[:,:2],np.ones(3)],t[:,2])
        out.append((Polygon(t[:,:2]),a))
    return out

def height_at(roof,x,y):
    from shapely.geometry import Point
    q=Point(x,y)
    zz=[float(np.dot(a,[x,y,1])) for p,a in roof if p.buffer(EPS).covers(q)]
    assert zz,(x,y)
    assert max(zz)-min(zz)<2e-4
    return float(np.mean(zz))

def rect_strip(a,b,c,d,roof,lift=.018):
    # Clip each flashing band against actual host roof triangles, so no
    # bridging across a gable ridge, or sinking into the opposite slope.
    q=box(a,b,c,d);out=[];area=0.
    for p,A in roof:
        hit=q.intersection(p)
        if hit.geom_type!="Polygon" or hit.area<1e-9:continue
        xy=np.array(hit.exterior.coords[:-1],float)
        # Intersection of two convex polygons is convex.
        v=np.c_[xy,np.c_[xy,np.ones(len(xy))]@A+lift]
        for j in range(1,len(v)-1):
            t=v[[0,j,j+1]]
            if np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))<1e-9:continue
            if np.cross(t[1]-t[0],t[2]-t[0])[2]<0:t=t[::-1]
            out.append(t)
        area+=hit.area
    assert abs(area-q.area)<1e-5,(q.bounds,area)
    assert out
    return np.array(out)

def ring_rect(lo,hi,margin,roof):
    a,b=lo[:2];c,d=hi[:2];m=margin
    rects=[(a-m,b-m,c+m,b),(a-m,d,c+m,d+m),(a-m,b,a,d),(c,b,c+m,d)]
    return np.concatenate([rect_strip(*r,roof) for r in rects])

def disk(x,y,z,r,n=12):
    rim=np.array([(x+r*np.cos(2*np.pi*i/n),y+r*np.sin(2*np.pi*i/n),z) for i in range(n)])
    return np.array([[[x,y,z],rim[i],rim[(i+1)%n]] for i in range(n)])

def plan(scene):
    roof,equip,frames=parts(scene); modules=[]; info={}; checks=0
    for b,tris in sorted(roof.items()):
        host=planes(tris); footprint=unary_union([p for p,_ in host])
        L,H=bbox(tris); flat=bool(H[2]-L[2]<EPS)
        # Occupancy excludes roof edge rings and balcony slabs, which are
        # not roof equipment. Guard is unchanged and outside inner zone.
        relevant={p:d for p,d in equip[b].items() if any(k in d for k in ("access","chimney","vent","dormer_front"))}
        bounds={p:bbox(np.concatenate([v for vs in d.values() for v in vs])) for p,d in relevant.items()}
        flash=[];drains=[];patches=[]
        def add(kind,key,geom,anchor=None):
            nonlocal checks
            lo,hi=bbox(np.concatenate([p.reshape(-1,3) for _,_,p in geom]))
            assert footprint.buffer(EPS).covers(box(*lo[:2],*hi[:2])),(kind,key,lo,hi)
            for other,(l,h) in bounds.items():
                if other==anchor:continue
                assert not np.all(np.minimum(hi,h)-np.maximum(lo,l)>EPS),(kind,key,other)
                checks+=1
            for other in modules:
                if other["building"]!=b:continue
                l,h=other["bounds"]
                # Conservative AABB for additions, zero-volume contacts OK.
                if np.all(np.minimum(hi,h)-np.maximum(lo,l)>EPS):
                    # Flashing spans multiple roof planes, so its aggregate
                    # bounding box is not a solid. Refine to individual
                    # triangles for an otherwise false dormer-sill clash.
                    for _,_,p in geom:
                        for _,_,q in other["geometry"]:
                            for pt in p:
                                a,c=bbox(pt)
                                for qt in q:
                                    A,C=bbox(qt)
                                    assert not np.all(np.minimum(c,C)-np.maximum(a,A)>EPS),(key,other["id"])
                checks+=1
            for k,mat,p in geom:
                assert np.isfinite(p).all()
                assert np.all(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)>1e-9)
            modules.append(dict(id=b+":stage24:"+key,kind=kind,building=b,anchor=anchor,
                geometry=geom,bounds=[lo.tolist(),hi.tolist()]))
        for p,d in sorted(relevant.items()):
            key=p.split(":")[-1]
            body=next(k for k in ("access","chimney","vent","dormer_front") if k in d)
            bodyparts=[v for k,vs in d.items() if k in ("access","access_front","chimney","vent","dormer_front","dormer_wall","dormer_base") for v in vs]
            lo,hi=bbox(np.concatenate(bodyparts))
            # Flashings contact the base perimeter only. Verify each band
            # is on its actual support plane with a small rendering offset.
            g=ring_rect(lo,hi,.13,host)
            for x,y,z in g.reshape(-1,3):
                assert abs(z-height_at(host,x,y)-.018)<2e-5
            add("flashing",key+":flashing",[("roof24_flashing","painted_rail",g)],p)
            flash.append([*lo[:2],*hi[:2]])
            if body=="chimney":
                l,h=bbox(np.concatenate(d["cap"])); x,y=(l[:2]+h[:2])/2;z=h[2]
                # Two flues on the existing masonry cap; no fake emission.
                for i,dx in enumerate((-.20,.20)):
                    t=tube(x+dx,y,z,.135,.26,8)
                    a,B=bbox(t);assert np.all(a[:2]>=l[:2]-EPS) and np.all(B[:2]<=h[:2]+EPS)
                    add("flue",key+":flue"+str(i),[("roof24_flue","painted_rail",t)],p)
            elif body=="vent":
                l,h=bbox(np.concatenate(d["cap"])); x,y=(l[:2]+h[:2])/2;z=h[2]
                w=min(.60,(h[0]-l[0])*.78);dep=min(.58,(h[1]-l[1])*.78);gap=.19
                geom=[]
                # Open-ended cowl, supported on two opaque side plates.
                for xx in (x-w/2,x+w/2-.055):
                    geom.append(("roof24_cowl","painted_rail",solid_box([xx,y-dep/2,z],[xx+.055,y+dep/2,z+gap])))
                geom.append(("roof24_cowl","painted_rail",solid_box([x-w/2-.035,y-dep/2-.035,z+gap],[x+w/2+.035,y+dep/2+.035,z+gap+.055])))
                a,B=bbox(np.concatenate([t.reshape(-1,3) for _,_,t in geom]))
                assert np.all(a[:2]>=l[:2]-EPS) and np.all(B[:2]<=h[:2]+EPS)
                add("cowl",key+":cowl",geom,p)
            elif body=="dormer_front":
                l,h=bbox(np.concatenate(d["dormer_front"]));x=(l[0]+h[0])/2
                y=l[1];sign=-1 if y<(L[1]+H[1])/2 else 1
                z=l[2]+.46*(h[2]-l[2])-.045
                a,bx=x-.48,x+.48; ya,yb=sorted((y,y+sign*.13))
                assert z>max(height_at(host,xx,yy) for xx in (a,bx) for yy in (ya,yb))+.04
                add("dormer_sill",key+":sill",[("roof24_sill","painted_trim",solid_box([a,ya,z],[bx,yb,z+.045]))],p)
        if flat:
            # Visible internal rainwater outlet proxies on the existing
            # flat plane. No claim of actual roof falls or hydraulic sizing.
            for i,(fx,fy) in enumerate(((.30,.23),(.72,.76))):
                x=L[0]+fx*(H[0]-L[0]);y=L[1]+fy*(H[1]-L[1]);z=height_at(host,x,y)
                if any(box(x-.45,y-.45,x+.45,y+.45).intersects(box(*a[:2],*c[:2])) for a,c in bounds.values()):continue
                add("drain","drain"+str(i),[("roof24_drain","painted_rail",disk(x,y,z+.022,.24))])
                drains.append([x,y])
            # Rectangular membrane repair panels, independent of UV origin.
            for fx,fy in ((.17,.73),(.49,.29),(.83,.46)):
                x=L[0]+fx*(H[0]-L[0]);y=L[1]+fy*(H[1]-L[1])
                q=box(x-.65,y-.4,x+.65,y+.4)
                if not any(q.intersects(box(*a[:2],*c[:2]).buffer(.18)) for a,c in bounds.values()):
                    patches.append(list(q.bounds))
        info[b]=dict(bounds=[L.tolist(),H.tolist()],flat=flat,flash=flash,drains=drains,patches=patches)
    assert len({m["id"] for m in modules})==len(modules)
    return modules,info,frames,checks
