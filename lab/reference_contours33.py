"""Measured contour tiers: section linework, polygons and independent extrusion.
All coordinates normalized to donor bounds; arbitrary footprint adaptation is
not implied. Open slices fall back explicitly, never called exact GIS.
"""
import numpy as np
from shapely import set_precision,constrained_delaunay_triangles,orient_polygons
from shapely.geometry import LineString,Polygon,MultiPolygon,box,mapping,shape
from shapely.ops import unary_union,polygonize

def section_lines(t,z,span):
    v=t[(t[:,:,2].min(1)<z)&(t[:,:,2].max(1)>z)]
    if not len(v):return []
    pts=[];ids=[]
    for i,j in ((0,1),(1,2),(2,0)):
        a,b=v[:,i],v[:,j];hit=(a[:,2]-z)*(b[:,2]-z)<0
        ii=np.flatnonzero(hit);a,b=a[hit],b[hit]
        q=a[:,:2]+(b[:,:2]-a[:,:2])*((z-a[:,2])/(b[:,2]-a[:,2]))[:,None]
        pts.append(q/span[:2]);ids.append(ii)
    p=np.concatenate(pts);idx=np.concatenate(ids)
    if len(p)<4:return []
    order=np.argsort(idx,kind='stable');p=p[order];idx=idx[order]
    if len(p)%2 or not np.array_equal(idx[::2],idx[1::2]):return []
    return p.reshape(-1,2,2)

def components(p):
    if p.geom_type=='Polygon':return [p]
    if p.geom_type in ('MultiPolygon','GeometryCollection'):return [q for g in p.geoms for q in components(g)]
    return []

def section_polygon(t,z,span,tol=.0015):
    return polygon_from_lines(section_lines(t,z,span),tol)

def polygon_from_lines(raw,tol=.0015):
    lines=[]
    for p in raw:
        if np.linalg.norm(p[1]-p[0])<tol*.35:continue
        g=set_precision(LineString(p),tol)
        if not g.is_empty and g.length>0:lines.append(g)
    if not lines:return None
    joined=unary_union(lines)
    polys=sorted((p for p in polygonize(joined) if p.area>=.0005),key=lambda p:-p.area)
    if not polys:return None
    # Keep exterior cycles; nested cycles are not separate occupied components.
    kept=[]
    for p in polys:
        if any(Polygon(q.exterior).covers(p.representative_point()) for q in kept):continue
        kept.append(p)
    result=unary_union(kept).simplify(.003,preserve_topology=True)
    if result.is_empty or not result.is_valid:return None
    # Reject closures that cover only a tiny minority of observed perimeter.
    observed=joined.length
    overlap=result.boundary.buffer(tol*2).intersection(joined).length
    if overlap/max(observed,1e-12)<.30:return None
    return result

def iou(a,b):
    return a.intersection(b).area/max(a.union(b).area,1e-12)

def medoid(group):
    # Candidate must be an observed slice, not a fictional union of all slices.
    if len(group)==1:return group[0][1]
    sub=group[::max(1,len(group)//5)]
    return max(sub,key=lambda x:sum(iou(x[1],q) for _,q in sub))[1]

def cap_evidence(t,n,a,span):
    """Cumulative observed horizontal face projection above each level.
    This is supporting evidence, not proof of enclosed occupancy."""
    from shapely import polygons,union_all
    ok=(abs(n[:,2])>.995)&(a>span[0]*span[1]*.000015)
    if not np.any(ok):return []
    tt=t[ok];zz=np.round(tt[:,:,2].mean(1)/span[2],4)
    levels=[];acc=None
    for z in sorted(set(zz),reverse=True):
        xy=tt[zz==z,:,:2]/span[:2]
        p=union_all(polygons(xy),grid_size=.0015)
        p=unary_union([q for q in components(p) if q.area>=.0005])
        if p.is_empty:continue
        acc=p if acc is None else acc.union(p)
        acc=acc.simplify(.003,preserve_topology=True)
        levels.append((z,acc))
    return levels

def supported_slice(t,vertical,z,span,levels,raw_p,r):
    # Gather vertical wall observations from nearby heights to avoid window
    # openings turning a wall into a disconnected or collapsed entire floor.
    dz=span[2]*.019
    raw=[section_lines(vertical,z+delta,span) for delta in (-dz,-dz*.5,0,dz*.5,dz)]
    lines=[x for x in raw if len(x)]
    multi=polygon_from_lines(np.concatenate(lines)) if lines else None
    p=raw_p
    if multi is not None and (p is None or (multi.area>p.area*1.07 and multi.intersection(p).area>=p.area*.55)):p=multi
    above=[g for h,g in levels if h>=z/span[2]-.0001]
    cap=above[-1] if above else None
    if cap is not None and r is not None:
        rect=box(*np.clip(r/np.tile(span[:2],2),0,1))
        if rect.area>0 and cap.area>=rect.area*.10:
            b=box(*cap.bounds);extent_iou=iou(b,rect)
            if p is None and extent_iou>=.35:p=cap
            elif p is not None and cap.area>p.area*1.12 and extent_iou>=.35 and cap.intersection(p).area>=p.area*.45:
                p=cap.union(p)
    return p.simplify(.004,preserve_topology=True) if p is not None else None

def fit_contours(t,n,a,span,base_tiers,count=48,repair_open=False):
    if min(span)<=1e-9:return [],0
    samples=[];valid=0
    from reference_match33 import slice_box
    vertical=t[abs(n[:,2])<.10] if repair_open else None
    levels=cap_evidence(t,n,a,span) if repair_open else []
    for i in range(count):
        z=(i+.473)*span[2]/count;p=section_polygon(t,z,span)
        if p is not None:valid+=1
        r=slice_box(t,z) if p is None or repair_open else None
        if repair_open:p=supported_slice(t,vertical,z,span,levels,p,r)
        if p is not None: samples.append((i,p));continue
        # Fallback is explicit: local slice bounds, never claim closed contour.
        if r is not None:
            r=np.clip(r/np.tile(span[:2],2),0,1)
            if min(r[2:]-r[:2])>=.06:samples.append((i,box(*r)))
    if len(samples)<count*.65:return [],valid
    groups=[]
    for i,p in samples:
        if groups and iou(p,groups[-1][-1][1])>=.90:groups[-1].append((i,p))
        else:groups.append([(i,p)])
    # Keep an observed terminal roof core even when only one sample thick.
    while len(groups)>1:
        small=[i for i,g in enumerate(groups) if len(g)<2 and i!=len(groups)-1]
        if len(groups)>10:small=list(range(len(groups)))
        if not small:break
        i=min(small,key=lambda j:(len(groups[j]),j))
        ns=[j for j in (i-1,i+1) if 0<=j<len(groups)]
        j=max(ns,key=lambda j:iou(medoid(groups[i]),medoid(groups[j])))
        lo,hi=sorted((i,j));groups[lo:hi+1]=[groups[lo]+groups[hi]]
    horiz=(abs(n[:,2])>.995)&(a>span[0]*span[1]*.0003)
    levels=np.unique(np.round(t[horiz,:,2].mean(1)/span[2],5))
    tiers=[];start=0.
    for i,g in enumerate(groups):
        end=1. if i==len(groups)-1 else (g[-1][0]+groups[i+1][0][0]+1)/2/count
        if i<len(groups)-1 and len(levels):
            z=float(levels[np.argmin(abs(levels-end))])
            if abs(z-end)<1.1/count and start+.003<z<1.-.003:end=z
        if end<=start+.001:continue
        p=medoid(g).simplify(.004,preserve_topology=True)
        tiers.append(dict(z0=start,z1=end,polygon=mapping(p),samples=len(g)))
        start=end
    if tiers:tiers[-1]['z1']=1.
    return tiers,valid

def mesh_contours(tiers,span,tolerance=.004):
    """Independent extrusion of measured polygon sections; no donor triangles."""
    from shapely.affinity import scale
    out=[];uv=[];role=[]
    def add(t,u,k):out.append(np.asarray(t,float));uv.append(np.asarray(u,float));role.append(k)
    def cap(p,z,up):
        for part in components(p):
            if part.area<=1e-10:continue
            for g in constrained_delaunay_triangles(part).geoms:
                xy=np.asarray(g.exterior.coords)[:3];t=np.c_[xy,np.full(3,z)]
                if abs(np.cross(t[1]-t[0],t[2]-t[0])[2])<1e-12:continue
                if (np.cross(t[1]-t[0],t[2]-t[0])[2]>0)!=up:t=t[::-1]
                add(t,t[:,:2]/span[:2],0)
    polys=[scale(shape(q['polygon']).simplify(tolerance,preserve_topology=True),xfact=span[0],yfact=span[1],origin=(0,0)) for q in tiers]
    for i,(q,p) in enumerate(zip(tiers,polys)):
        z,Z=q['z0']*span[2],q['z1']*span[2]
        for part in components(orient_polygons(p)):
            for ring in [part.exterior,*part.interiors]:
                xy=np.asarray(ring.coords)
                for a,b in zip(xy[:-1],xy[1:]):
                    w=np.linalg.norm(b-a)
                    if w<=1e-9:continue
                    v=np.array([[*a,z],[*b,z],[*b,Z],[*a,Z]])
                    u=np.array([[0,z],[w,z],[w,Z],[0,Z]])
                    for f in ([0,1,2],[0,2,3]):add(v[f],u[f],1)
        cap(p if i==0 else p.difference(polys[i-1]),z,False)
        cap(p if i==len(polys)-1 else p.difference(polys[i+1]),Z,True)
    return np.array(out),np.array(uv),np.array(role)

def bounded_mesh(tiers,span,cap=1800):
    for tol in (.004,.008,.015,.025,.04):
        t,u,r=mesh_contours(tiers,span,tol)
        if 0<len(t)<=cap:return t,u,r,tol
    return None
