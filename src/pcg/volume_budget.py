"""Deterministic geometry reservation for opt-in volume grammar.
Costs are upper bounds for whole canonical modules, not measured GPU cost.
The optional detail planner separately enforces the hard mesh cap.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Offer:
    id: str
    triangles: int
    priority: int
    tie: int

def select(offers, capacity):
    if type(capacity) is not int or capacity < 0:
        raise ValueError('invalid capacity')
    offers=tuple(offers)
    if len({x.id for x in offers}) != len(offers):
        raise ValueError('duplicate geometry offer')
    if any(type(x.triangles) is not int or x.triangles<=0 for x in offers):
        raise ValueError('invalid geometry cost')
    accepted=[];used=0
    for offer in sorted(offers,key=lambda x:(-x.priority,x.tie,x.id)):
        if used+offer.triangles<=capacity:
            accepted.append(offer.id);used+=offer.triangles
    return tuple(accepted),used

def detail_reserve(rings,entries,roof):
    # Explicit allowance for roof edges, entry plates, equipment and trim.
    # This is a conservative planning allowance, not a promise all optional
    # detail fits: the mesh allocator can reject whole secondary modules.
    edges=sum(map(len,rings))
    return min(800,max(300,120+edges*(24 if roof=='flat' else 30)+24*entries))
