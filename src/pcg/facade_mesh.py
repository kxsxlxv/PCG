"""Facade render triangles with actual wall apertures; semantic massing remains intact."""
from dataclasses import dataclass
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from .architecture import Facade


@dataclass(frozen=True)
class FacadeTriangle:
    role: str
    opening_id: str | None
    vertices: tuple


def _rect(o):
    return ((o.u,o.z),(o.u+o.width,o.z),(o.u+o.width,o.z+o.height),(o.u,o.z+o.height))


def facade_triangles(f:Facade,height:float,lod:int=0):
    """LOD0 recessed panels; LOD1 flush panels from the same plan.
    The wall has a real cutout at both LODs. No interior simulation.
    """
    if type(lod) is not int or lod not in (0,1):
        raise ValueError("unsupported facade LOD")
    outer=box(0,0,f.length,height)
    holes=[Polygon(_rect(o)) for o in f.openings]
    if any(not outer.contains(p) or outer.boundary.intersects(p) for p in holes):
        raise ValueError("opening outside strict wall bounds")
    if any(a.intersects(b) for i,a in enumerate(holes) for b in holes[:i]):
        raise ValueError("overlapping openings")
    wall=Polygon(outer.exterior.coords,[p.exterior.coords for p in holes])
    if not wall.is_valid:
        raise ValueError("invalid aperture wall")
    triangles=list(constrained_delaunay_triangles(wall).geoms)
    if abs(sum(t.area for t in triangles)-wall.area)>1e-8*max(1,wall.area) or unary_union(triangles).symmetric_difference(wall).area>1e-8*max(1,wall.area):
        raise ValueError("aperture CDT coverage")
    result=[]
    for t in triangles:
        pts=list(t.exterior.coords)[:-1]
        if not t.exterior.is_ccw:pts.reverse()
        j=pts.index(min(pts))
        pts=pts[j:]+pts[:j]
        result.append(FacadeTriangle("wall",None,tuple(f.point(*p) for p in pts)))
    result.sort(key=lambda t:t.vertices)
    def quad(role,oid,p):
        result.extend(FacadeTriangle(role,oid,tuple(p[i] for i in idx)) for idx in ((0,1,2),(0,2,3)))
    for o in f.openings:
        uv=_rect(o)
        front=[f.point(*p) for p in uv]
        back=[f.point(*p,depth=o.recess if lod==0 else 0.) for p in uv]
        quad(o.role,o.id,back)
        if lod==0:
            for i in range(4):
                j=(i+1)%4
                quad("reveal",o.id,(front[i],front[j],back[j],back[i]))
    return tuple(result)
