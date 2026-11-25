"""Polygon massing: constrained caps, analytic convex hip, explicit fallback."""
from collections import defaultdict
from dataclasses import dataclass
import math
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon
from shapely.ops import unary_union
from .contracts import number, Rectangle
from .footprint import Footprint
from .geometry import Shell, Surface, Edge, build_shell


@dataclass(frozen=True)
class Massing:
    shell: Shell
    requested_roof: str
    actual_roof: str
    fallback_reason: str | None = None


def repair_duplicates(building_id, exterior, holes=()):
    """Only remove consecutive identical XY points; never change topology."""
    count = 0
    def clean(ring):
        nonlocal count
        out = []
        for p in ring:
            if not isinstance(p, (tuple, list)) or len(p) != 2:
                raise ValueError("expected XY coordinate")
            p = tuple(number(x, "coordinate") for x in p)
            if out and out[-1] == p:
                count += 1
            else:
                out.append(p)
        return out
    f = Footprint(building_id, clean(exterior), [clean(h) for h in holes])
    return f, {"removed_consecutive_duplicates": count, "area_delta_m2": 0.0,
               "method": "exact_duplicate_removal"}


def _caps(f, xy):
    idx = {p: i for i, p in enumerate(xy)}
    poly = f.polygon()
    pieces = list(constrained_delaunay_triangles(poly).geoms)
    if not pieces or any(not poly.covers(t) for t in pieces):
        raise ValueError("CDT outside footprint")
    union = unary_union(pieces)
    if not math.isclose(union.area, poly.area, rel_tol=1e-10) or not math.isclose(
            sum(t.area for t in pieces), poly.area, rel_tol=1e-10):
        raise ValueError("CDT coverage or overlap")
    tris = []
    for t in pieces:
        coords = list(t.exterior.coords)[:-1]
        if len(coords) != 3 or t.area <= 0:
            raise ValueError("invalid CDT triangle")
        if not t.exterior.is_ccw:
            coords.reverse()
        ids = [idx[p] for p in coords]
        i = ids.index(min(ids))
        tris.append(tuple(ids[i:] + ids[:i]))
    return sorted(tris)


def _assemble(bid, verts, specs, hip=False):
    surfaces = tuple(Surface(bid + ":" + name, role, tuple(ids))
                     for name, role, ids in specs)
    incidence = defaultdict(list)
    for s in surfaces:
        for a,b in zip(s.vertices,s.vertices[1:]+s.vertices[:1]):
            incidence[tuple(sorted((a,b)))].append((s,a,b))
    edges = []
    for (a,b), pair in sorted(incidence.items()):
        if len(pair) != 2 or pair[0][1:] != pair[1][1:][::-1]:
            raise ValueError("nonmanifold or inconsistent polygon shell")
        roles = {p[0].role for p in pair}
        role = ("seam" if roles == {"base"} else
                ("ridge" if math.isclose(verts[a][2],verts[b][2],rel_tol=0.,abs_tol=1e-9) else "hip") if roles == {"roof"} and hip else
                "seam" if roles == {"roof"} else
                "foundation" if "base" in roles else
                "eave" if "roof" in roles else "corner")
        edges.append(Edge(f"{bid}:edge:{a}:{b}",(a,b),
                          tuple(p[0].id for p in pair),role))
    return Shell(bid,tuple(verts),surfaces,tuple(edges))


def _flat(f,h):
    rings = (f.exterior, *f.holes)
    xy = tuple(p for ring in rings for p in ring)
    n = len(xy)
    verts = [(x,y,z) for z in (0.,h) for x,y in xy]
    caps = _caps(f,xy)
    specs = [(f"base:{i}","base",t[::-1]) for i,t in enumerate(caps)]
    start = 0
    for r,ring in enumerate(rings):
        for i in range(len(ring)):
            a,b = start+i,start+(i+1)%len(ring)
            specs.append((f"wall:{r}:{i}","wall",(a,b,b+n,a+n)))
        start += len(ring)
    specs += [(f"roof:{i}","roof",tuple(j+n for j in t)) for i,t in enumerate(caps)]
    return _assemble(f.building_id,verts,specs)


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])


def _rectangle_gable(f,h,rise):
    if f.holes or len(f.exterior) != 4:
        return None
    a,b,c,d = f.exterior
    u=(b[0]-a[0],b[1]-a[1]); v=(d[0]-a[0],d[1]-a[1])
    w,l=math.hypot(*u),math.hypot(*v)
    if abs(u[0]*v[0]+u[1]*v[1]) > 1e-12*w*l or math.dist(c,(b[0]+v[0],b[1]+v[1])) > 1e-12*max(w,l):
        return None
    # Ridge follows the longer edge, including rotated rectangles.
    if l > w:
        a,b,c,d=b,c,d,a
        u=(b[0]-a[0],b[1]-a[1]); v=(d[0]-a[0],d[1]-a[1])
        w,l=l,w
    s=build_shell(Rectangle(f.building_id,w,l,h,"gable",rise))
    verts=[(a[0]+x*u[0]/w+y*v[0]/l,a[1]+x*u[1]/w+y*v[1]/l,z) for x,y,z in s.vertices]
    verts[:4]=[(x,y,0.) for x,y in (a,b,c,d)]
    verts[4:8]=[(x,y,h) for x,y in (a,b,c,d)]
    return Shell(s.building_id,tuple(verts),s.surfaces,s.edges)


def _hip(f,h,rise):
    # Equal-pitch convex roof = lower envelope of inward edge-distance planes.
    # Strictly convex, no holes; collinear and concave inputs fall back.
    ring=f.exterior
    if f.holes or any(_cross(ring[i-1],ring[i],ring[(i+1)%len(ring)]) <= 0 for i in range(len(ring))):
        return None
    ox,oy=ring[0]
    pts=[(x-ox,y-oy) for x,y in ring]
    scale=max(math.hypot(*p) for p in pts)
    eps=scale*1e-10
    planes=[]
    for a,b in zip(pts,pts[1:]+pts[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        l=math.hypot(dx,dy)
        nx,ny=-dy/l,dx/l
        planes.append((nx,ny,-nx*a[0]-ny*a[1]))
    def val(p,q):return p[0]*q[0]+p[1]*q[1]+q[2]
    def clip(poly,q):
        out=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            va,vb=val(a,q),val(b,q)
            ia,ib=va<=0,vb<=0
            if ia:out.append(a)
            if ia!=ib:
                t=va/(va-vb)
                out.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
        return out
    nodes=list(pts)
    faces=[]
    for i,p in enumerate(planes):
        cell=list(pts)
        for j,q in enumerate(planes):
            if i!=j and cell:cell=clip(cell,tuple(a-b for a,b in zip(p,q)))
        ids=[]
        for q in cell:
            match=next((k for k,v in enumerate(nodes) if math.dist(q,v)<=eps),None)
            if match is None:
                match=len(nodes);nodes.append(q)
            if not ids or ids[-1]!=match:ids.append(match)
        if len(ids)>1 and ids[0]==ids[-1]:ids.pop()
        if len(ids)<3 or len(set(ids))!=len(ids):return None
        faces.append(ids)
    # Split every roof boundary at shared nodes, avoiding T-junctions.
    split=[]
    for ids in faces:
        expanded=[]
        for a,b in zip(ids,ids[1:]+ids[:1]):
            p,q=nodes[a],nodes[b]
            dx,dy=q[0]-p[0],q[1]-p[1]; l2=dx*dx+dy*dy
            mids=[]
            for k,r in enumerate(nodes):
                if k in (a,b):continue
                t=((r[0]-p[0])*dx+(r[1]-p[1])*dy)/l2
                if 0<t<1 and math.dist(r,(p[0]+t*dx,p[1]+t*dy))<=eps:
                    mids.append((t,k))
            expanded.extend([a]+[k for _,k in sorted(mids)])
        split.append(expanded)
    n=len(ring)
    heights=[0. if i<n else min(val(p,q) for q in planes) for i,p in enumerate(nodes)]
    peak=max(heights)
    if peak<=eps or any(z < -eps for z in heights):return None
    verts=[(x,y,0.) for x,y in ring]+[(x+ox,y+oy,h+rise*z/peak) for (x,y),z in zip(nodes,heights)]
    verts[n:2*n]=[(x,y,h) for x,y in ring]
    specs=[(f"base:{i}","base",t[::-1]) for i,t in enumerate(_caps(f,ring))]
    for i in range(n):
        j=(i+1)%n
        specs.append((f"wall:{i}","wall",(i,j,j+n,i+n)))
    # Existing Shell fan requires nondegenerate triangles. Reject rare cells
    # whose collinear vertices cannot be represented safely by a single fan.
    for i,ids in enumerate(split):
        for k in range(len(ids)):
            ordered=ids[k:]+ids[:k]
            if all(_cross(nodes[ordered[0]],nodes[ordered[j]],nodes[ordered[j+1]])>eps*eps for j in range(1,len(ids)-1)):
                break
        else:return None
        specs.append((f"roof:{i}","roof",tuple(j+n for j in ordered)))
    try:
        return _assemble(f.building_id,verts,specs,hip=True)
    except ValueError:
        return None


def build_massing(f: Footprint, eave_m, roof="flat", roof_rise_m=0.) -> Massing:
    h=number(eave_m,"eave_m"); rise=number(roof_rise_m,"roof_rise_m")
    if h<=0 or rise<0 or (roof=="flat" and rise!=0) or (roof in ("gable","hip") and rise<=0):
        raise ValueError("inconsistent height/roof")
    if roof=="flat":return Massing(_flat(f,h),roof,roof)
    if roof=="gable":
        s=_rectangle_gable(f,h,rise)
        if s is not None:return Massing(s,roof,roof)
        reason="gable_requires_rectangle_without_holes"
    elif roof=="hip":
        s=_hip(f,h,rise)
        if s is not None:return Massing(s,roof,roof)
        reason="hip_requires_supported_strictly_convex_footprint"
    else:
        raise ValueError("unknown roof type")
    return Massing(_flat(f,h),roof,"flat",reason)
