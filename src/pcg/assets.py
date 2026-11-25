"""Reusable metric facade modules and stable placement plans.

Box geometry uses the audited upstream primitive. Fit, orientation, LOD and
cross-module exclusion policy belong to PCG. Local +X tangent, -Y outward,
+Z up; all dimensions in meters.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from ._grammar_boxes import _oriented_box
from .architecture import ArchitecturePlan
from .footprint import Footprint
from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass(frozen=True)
class Part:
    role: str
    center: tuple
    size: tuple


@dataclass(frozen=True)
class Recipe:
    family: str
    width: float
    height: float
    recess: float
    revision: int = 1

    @property
    def id(self):
        return self.family+":"+hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(",",":")).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Placement:
    id: str
    facade_id: str
    opening_id: str
    asset_id: str
    origin: tuple
    tangent: tuple
    outward: tuple

    def point(self,p):
        x,y,z=p
        return (self.origin[0]+x*self.tangent[0]-y*self.outward[0],
                self.origin[1]+x*self.tangent[1]-y*self.outward[1],
                self.origin[2]+z)


@dataclass(frozen=True)
class AssetPlan:
    architecture_digest: str
    recipes: tuple[Recipe,...]
    placements: tuple[Placement,...]
    rejected: tuple[tuple[str,str],...]


def parts(recipe,lod):
    if type(lod) is not int or lod not in (0,1):
        raise ValueError("unsupported LOD")
    w,h,r=recipe.width,recipe.height,recipe.recess
    t=min(.065,w/8,h/8)
    # Recess frame projects toward exterior from panel, but stays within
    # aperture XY extent. At LOD1 mount at facade plane.
    y=r-.018 if lod==0 else -.018
    if recipe.family in ("window","door"):
        role=recipe.family+"_frame"
        p=[Part(role,(x,y,h/2),(t,.036,h)) for x in (-w/2+t/2,w/2-t/2)]
        p.extend(Part(role,(0,y,z),(w-2*t,.036,t)) for z in (t/2,h-t/2))
        if recipe.family=="window" and lod==0:
            p.append(Part("window_mullion",(0,y,h/2),(t*.7,.036,h-2*t)))
        if recipe.family=="door" and lod==0:
            p.append(Part("door_handle",(w*.28,y-.035,h*.52),(.04,.06,.10)))
        return tuple(p)
    if recipe.family=="balcony":
        d=.75; t=.06
        p=[Part("balcony_slab",(0,-d/2,.06),(w,d,.12)),
           Part("balcony_rail",(0,-d+t/2,1.04),(w,t,.08)),
           Part("balcony_rail",(-w/2+t/2,-(d-t)/2,.59),(t,d-t,.94)),
           Part("balcony_rail",(w/2-t/2,-(d-t)/2,.59),(t,d-t,.94))]
        if lod==0:
            for i in range(1,6):
                p.append(Part("balcony_bar",(-w/2+w*i/6,-d+t/2,.56),(.035,t,.88)))
        else:
            p.append(Part("balcony_panel",(0,-d+t/2,.56),(w,t,.88)))
        return tuple(p)
    raise ValueError("unknown family")


def asset_triangles(recipe,lod):
    """Each part is a closed box; no union across intentional contacts."""
    out=[]
    for p in parts(recipe,lod):
        x,y,z=p.center; w,d,h=p.size
        verts,faces=_oriented_box((x,y),(1.,0.),(0.,-1.),w,d,h,z-h/2)
        for f in faces:
            for i in range(1,len(f)-1):
                out.append((p.role,tuple(verts[j] for j in (f[0],f[i],f[i+1]))))
    return tuple(out)


def plan_assets(f:Footprint,plan:ArchitecturePlan):
    if f.digest()!=plan.footprint_digest:
        raise ValueError("footprint mismatch")
    recipes={}; placements=[]; rejected=[]
    balconies=[]
    poly=f.polygon()
    def add(s,o,recipe,origin):
        recipes[recipe.id]=recipe
        tangent=((s.end[0]-s.start[0])/s.length,(s.end[1]-s.start[1])/s.length)
        placements.append(Placement(o.id+":"+recipe.family,s.id,o.id,recipe.id,origin,tangent,s.outward))
    for s in plan.facades:
        if s.party:
            continue
        for o in s.openings:
            u=o.u+o.width/2
            add(s,o,Recipe(o.role,o.width,o.height,o.recess),s.point(u,o.z))
            # Illustrative apartment slab/rail module only on upper exterior
            # bays, with a clear margin to adjacent walls and other modules.
            if plan.profile.id!="apartments" or s.courtyard or o.role!="window" or o.floor==0:
                continue
            w=o.width+.3; z=plan.floor_levels[o.floor]+.05
            # Projected balcony must lie outside the footprint, fit within
            # facade length and avoid other balconies at overlapping heights.
            if u-w/2<.15 or u+w/2>s.length-.15 or z+1.08>plan.floor_levels[o.floor+1]:
                rejected.append((o.id,"balcony_fit"));continue
            q=Polygon([s.point(x,0,depth)[:2] for x,depth in
                       ((u-w/2,0),(u+w/2,0),(u+w/2,-.75),(u-w/2,-.75))])
            if q.intersection(poly).area>1e-9 or any(q.intersection(other).area>1e-9 and z<top and z+1.08>bottom for other,bottom,top in balconies):
                rejected.append((o.id,"balcony_intersection"));continue
            balconies.append((q,z,z+1.08))
            add(s,o,Recipe("balcony",w,1.08,0.),s.point(u,z))
    return AssetPlan(plan.digest(),tuple(recipes[k] for k in sorted(recipes)),
                     tuple(placements),tuple(rejected))
