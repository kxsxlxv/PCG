"""Canonical facade layout; upstream bay packing, PCG bounds and exclusion policy."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from .contracts import number
from .footprint import Footprint
from ._grammar_bays import window_offsets
from shapely.geometry import Polygon

BALCONY_DEPTH = .75
BALCONY_WIDTH_EXTRA = .30
BALCONY_BASE = .05
BALCONY_SLAB = .12
BALCONY_HEIGHT = 1.08
BALCONY_THRESHOLD = BALCONY_BASE + BALCONY_SLAB



@dataclass(frozen=True)
class Profile:
    id: str
    window_width: float
    window_height: float
    sill: float
    spacing: float
    margin: float
    door_width: float
    door_height: float
    recess: float = .12
    revision: int = 1

    def __post_init__(self):
        if not isinstance(self.id,str) or not self.id or type(self.revision) is not int or self.revision != 1:
            raise ValueError("invalid profile identity")
        for k in ("window_width","window_height","sill","spacing","margin","door_width","door_height","recess"):
            v=number(getattr(self,k),k)
            if v<=0:raise ValueError(k+" must be positive")
            object.__setattr__(self,k,v)
        if self.spacing<=self.window_width:
            raise ValueError("window spacing must leave a pier")


PROFILES = {
    "house":Profile("house",1.25,1.45,.85,2.7,.8,1.0,2.1),
    "apartments":Profile("apartments",1.4,1.5,.85,2.8,.8,1.2,2.2),
    "office":Profile("office",2.2,1.9,.7,2.8,.6,1.6,2.25),
}


@dataclass(frozen=True)
class Opening:
    id: str
    role: str
    floor: int
    u: float
    z: float
    width: float
    height: float
    recess: float


@dataclass(frozen=True)
class Facade:
    id: str
    start: tuple
    end: tuple
    length: float
    outward: tuple
    courtyard: bool
    party: bool
    openings: tuple[Opening,...]

    def point(self,u,z,depth=0.):
        dx=(self.end[0]-self.start[0])/self.length
        dy=(self.end[1]-self.start[1])/self.length
        return (self.start[0]+dx*u-self.outward[0]*depth,
                self.start[1]+dy*u-self.outward[1]*depth,z)


@dataclass(frozen=True)
class ArchitecturePlan:
    building_id: str
    footprint_digest: str
    profile: Profile
    floor_levels: tuple[float,...]
    facades: tuple[Facade,...]

    def canonical(self):
        return json.dumps(asdict(self),sort_keys=True,separators=(",",":"),allow_nan=False)

    def digest(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()


def plan_facades(f:Footprint,eave_m,levels,profile:Profile,*,party_edges=(),entrance_edge=None):
    """Party edges and entrance refer to (canonical ring index, edge index).
    A suppressed party edge is entire facade, not automatic GIS adjacency.
    The door retains a positive threshold to keep its aperture inside the wall.
    """
    h=number(eave_m,"eave_m")
    if h<=0 or type(levels) is not int or not 1<=levels<=200:
        raise ValueError("invalid floor contract")
    if not isinstance(profile,Profile):raise ValueError("expected Profile")
    floor_h=h/levels
    floors=tuple(h*i/levels for i in range(levels+1))
    rings=(f.exterior,*f.holes)
    keys={(r,i) for r,ring in enumerate(rings) for i in range(len(ring))}
    party=set(party_edges)
    if not party<=keys:raise ValueError("unknown party edge")
    if entrance_edge is not None and (entrance_edge not in keys or entrance_edge in party or entrance_edge[0]!=0):
        raise ValueError("invalid entrance edge")
    if entrance_edge is None:
        eligible=[(math.dist(a,b),-i) for i,(a,b) in enumerate(zip(f.exterior,f.exterior[1:]+f.exterior[:1])) if (0,i) not in party]
        if eligible:entrance_edge=(0,-max(eligible)[1])
    facades=[]
    balconies=[]
    polygon=f.polygon()
    clearance=.05
    for r,ring in enumerate(rings):
        for i,(a,b) in enumerate(zip(ring,ring[1:]+ring[:1])):
            length=math.dist(a,b)
            normal=((b[1]-a[1])/length,-(b[0]-a[0])/length)
            fid=f"{f.building_id}:facade:{r}:{i}"
            suppressed=(r,i) in party
            openings=[]
            door=None
            # Explicit threshold keeps the opening strictly inside the facade polygon.
            if not suppressed and (r,i)==entrance_edge and length>=profile.door_width+2*profile.margin and floor_h>=profile.door_height+2*clearance:
                door=Opening(fid+":door", "door",0,(length-profile.door_width)/2,clearance,profile.door_width,profile.door_height,profile.recess)
                openings.append(door)
            if not suppressed and profile.sill+profile.window_height+clearance<=floor_h:
                centers=window_offsets(length,profile.window_width,profile.spacing,profile.margin)
                for floor in range(levels):
                    for bay,center in enumerate(centers):
                        u=center-profile.window_width/2
                        if floor==0 and door and u<door.u+door.width+clearance and u+profile.window_width>door.u-clearance:
                            continue
                        role="window"
                        bottom=floors[floor]+profile.sill
                        height=profile.window_height
                        # Select the access opening and balcony together. An
                        # infeasible balcony leaves an ordinary window, never
                        # an unguarded elevated door.
                        if profile.id=="apartments" and r==0 and floor>0:
                            width=profile.window_width+BALCONY_WIDTH_EXTRA
                            left,right=center-width/2,center+width/2
                            base=floors[floor]+BALCONY_BASE
                            top=base+BALCONY_HEIGHT
                            door_bottom=floors[floor]+BALCONY_THRESHOLD
                            door_height=profile.door_height
                            if left>=.15 and right<=length-.15 and top<=floors[floor+1] and door_bottom+door_height+clearance<=floors[floor+1]:
                                tx,ty=(b[0]-a[0])/length,(b[1]-a[1])/length
                                q=Polygon([(a[0]+x*tx+d*normal[0],a[1]+x*ty+d*normal[1]) for x,d in ((left,0),(right,0),(right,BALCONY_DEPTH),(left,BALCONY_DEPTH))])
                                if q.intersection(polygon).area<=1e-9 and not any(q.intersection(other).area>1e-9 and base<high and top>low for other,low,high in balconies):
                                    balconies.append((q,base,top))
                                    role="balcony_door"
                                    bottom=door_bottom
                                    height=door_height
                        openings.append(Opening(f"{fid}:window:{floor}:{bay}",role,floor,u,bottom,profile.window_width,height,profile.recess))
            facades.append(Facade(fid,a,b,length,normal,r>0,suppressed,tuple(openings)))
    return ArchitecturePlan(f.building_id,f.digest(),profile,floors,tuple(facades))
