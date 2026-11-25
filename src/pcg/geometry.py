"""Analytic rectangular shell; welded semantic topology, local metric Z-up."""
from dataclasses import dataclass
from collections import defaultdict
from .contracts import Rectangle

@dataclass(frozen=True)
class Surface:
    id: str
    role: str
    vertices: tuple[int, ...]

@dataclass(frozen=True)
class Edge:
    id: str
    vertices: tuple[int, int]
    surfaces: tuple[str, str]
    role: str

@dataclass(frozen=True)
class Shell:
    building_id: str
    vertices: tuple[tuple[float, float, float], ...]
    surfaces: tuple[Surface, ...]
    edges: tuple[Edge, ...]

    def triangles(self):
        return tuple((s.id, (s.vertices[0], s.vertices[i], s.vertices[i+1]))
                     for s in self.surfaces for i in range(1, len(s.vertices)-1))

def build_shell(b: Rectangle) -> Shell:
    w,d,h=b.width_m,b.depth_m,b.eave_m
    v=((0.,0.,0.),(w,0.,0.),(w,d,0.),(0.,d,0.),
       (0.,0.,h),(w,0.,h),(w,d,h),(0.,d,h))
    specs=[("base","base",(3,2,1,0)),("south","wall",(0,1,5,4)),
           ("north","wall",(2,3,7,6))]
    if b.roof=="flat":
        specs += [("east","wall",(1,2,6,5)),("west","wall",(3,0,4,7)),
                  ("roof","roof",(4,5,6,7))]
    else:
        v += ((0.,d/2,h+b.roof_rise_m),(w,d/2,h+b.roof_rise_m))
        specs += [("east","wall",(1,2,6,9,5)),("west","wall",(3,0,4,8,7)),
                  ("roof_south","roof",(4,5,9,8)),("roof_north","roof",(8,9,6,7))]
    # Rotate polygon start to avoid collinear fan triangles on gable end walls.
    surfaces=tuple(Surface(b.building_id+":"+name,role,ids) for name,role,ids in specs)
    incidence=defaultdict(list)
    for s in surfaces:
        for a,c in zip(s.vertices,s.vertices[1:]+s.vertices[:1]):
            incidence[tuple(sorted((a,c)))].append((s,a,c))
    edges=[]
    for (a,c),pair in sorted(incidence.items()):
        if len(pair)!=2 or pair[0][1:]!=pair[1][1:][::-1]:
            raise ValueError("nonmanifold or inconsistent shell")
        roles={p[0].role for p in pair}
        role=("foundation" if "base" in roles else "ridge" if roles=={"roof"}
              else ("gable" if max(a,c)>=8 else "eave") if "roof" in roles else "corner")
        edges.append(Edge(f"{b.building_id}:edge:{a}:{c}",(a,c),
                          tuple(p[0].id for p in pair),role))
    return Shell(b.building_id,v,surfaces,tuple(edges))
