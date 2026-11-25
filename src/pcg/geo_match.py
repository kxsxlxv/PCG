"""Cross-source building candidate graph; no identity assignment or merge.

Reuses STRtree and overlap criteria from research/01_geo/audit_building_data.py.
All thresholds are uncalibrated review rules, not probabilities.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import math
from shapely import from_wkb
from shapely.strtree import STRtree
from .geo_normalize import SourceFeature, digest


@dataclass(frozen=True, order=True)
class SourceKey:
    source_id: str
    feature_id: str


@dataclass(frozen=True)
class MatchPolicy:
    high_iou: float = .5
    review_iou: float = .3
    containment: float = .8
    gap_m: float = 3.

    def __post_init__(self):
        vals=(self.high_iou,self.review_iou,self.containment,self.gap_m)
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in vals):
            raise ValueError("nonfinite match policy")
        if not (0<self.review_iou<=self.high_iou<=1 and 0<self.containment<=1 and 0<=self.gap_m<=10):
            raise ValueError("invalid match policy")


@dataclass(frozen=True)
class Candidate:
    left: SourceKey
    right: SourceKey
    iou: float
    intersection_m2: float
    overlap_smaller: float
    centroid_distance_m: float
    gap_m: float
    rule: str


@dataclass(frozen=True)
class MatchGroup:
    group_id: str
    left: tuple[SourceKey,...]
    right: tuple[SourceKey,...]
    status: str


def key(f):
    return SourceKey(f.source_id,f.source_feature_id)


def _prepare(records):
    ordered=sorted(records,key=key)
    keys=[key(f) for f in ordered]
    if len(set(keys))!=len(keys):
        raise ValueError("duplicate source feature key")
    geometries=[]
    for f in ordered:
        g=from_wkb(bytes.fromhex(f.local_geometry_wkb))
        if g.geom_type not in ("Polygon","MultiPolygon") or g.is_empty or not g.is_valid or not math.isfinite(g.area) or g.area<=0:
            raise ValueError("invalid normalized geometry")
        if not math.isclose(g.area,f.area_m2,rel_tol=1e-9):
            raise ValueError("normalized area mismatch")
        geometries.append(g)
    return ordered,geometries


def build_graph(left,right,policy=MatchPolicy()):
    """Two distinct source layers, same projected world frame; parts excluded."""
    left=tuple(left);right=tuple(right)
    all_records=left+right
    if any(not isinstance(f,SourceFeature) for f in all_records):
        raise ValueError("expected normalized SourceFeature records")
    if len({f.world_frame_digest for f in all_records})>1:
        raise ValueError("world frame mismatch")
    if len({f.source_id for f in left})>1 or len({f.source_id for f in right})>1:
        raise ValueError("expected one source layer per side")
    if {f.source_id for f in left}&{f.source_id for f in right}:
        raise ValueError("source layers must be distinct")
    if any(f.geometry_kind not in ("building","part") for f in all_records):
        raise ValueError("unknown geometry kind")
    if any(len({f.snapshot_digest for f in side})>1 for side in (left,right)):
        raise ValueError("mixed snapshots on one side")
    if len({key(f) for f in all_records})!=len(all_records):
        raise ValueError("duplicate source feature key")
    excluded=tuple(sorted(key(f) for f in all_records if f.geometry_kind=="part"))
    a,ga=_prepare(f for f in left if f.geometry_kind=="building")
    b,gb=_prepare(f for f in right if f.geometry_kind=="building")
    edges=[];touches=[];weak=[]
    tree=STRtree(gb)
    for f,g in zip(a,ga):
        for j in sorted(map(int,tree.query(g,predicate="dwithin",distance=policy.gap_m))):
            h=gb[j]; k=key(b[j])
            inter=g.intersection(h).area
            gap=g.distance(h)
            iou=inter/(g.area+h.area-inter)
            smaller=inter/min(g.area,h.area)
            pair=(key(f),k)
            if inter<=0 and g.touches(h):
                touches.append(pair)
                continue
            if iou>=policy.high_iou:
                rule="high_iou"
            elif smaller>=policy.containment:
                rule="containment_review"
            elif iou>=policy.review_iou and (g.covers(h.centroid) or h.covers(g.centroid)):
                rule="centroid_review"
            elif inter<=0 and gap<=policy.gap_m:
                rule="proximity_review"
            else:
                weak.append(pair)
                continue
            edges.append(Candidate(*pair,float(iou),float(inter),float(smaller),
                                   float(g.centroid.distance(h.centroid)),float(gap),rule))
    adj=defaultdict(set)
    edge_by_pair={}
    for edge in edges:
        adj[edge.left].add(edge.right);adj[edge.right].add(edge.left)
        edge_by_pair[(edge.left,edge.right)]=edge
    akeys={key(f) for f in a};bkeys={key(f) for f in b}
    # Connected components are conflict groups only; never transitive identities.
    unseen=akeys|bkeys;groups=[]
    while unseen:
        root=min(unseen);pending=[root];component=set()
        while pending:
            current=pending.pop()
            if current in component:continue
            component.add(current)
            pending.extend(sorted(adj[current]-component))
        unseen-=component
        ls=tuple(sorted(component&akeys));rs=tuple(sorted(component&bkeys))
        if not ls or not rs:
            status="unmatched_left" if ls else "unmatched_right"
        elif len(ls)==len(rs)==1:
            edge=edge_by_pair[(ls[0],rs[0])]
            status="isolated_high_candidate" if edge.rule=="high_iou" else "review_1to1"
        else:
            status="ambiguous_1toN" if len(ls)==1 else "ambiguous_Nto1" if len(rs)==1 else "ambiguous_NtoM"
        gid=digest({"left":[asdict(k) for k in ls],"right":[asdict(k) for k in rs]})
        groups.append(MatchGroup(gid,ls,rs,status))
    groups.sort(key=lambda g:(g.left,g.right))
    return {"schema_version":1,"compiler_version":"0.1.0","policy":asdict(policy),
            "world_frame_digest":next(iter({f.world_frame_digest for f in all_records}),None),
            "snapshots":sorted({(f.source_id,f.snapshot_digest) for f in all_records}),
            "edges":[asdict(e) for e in edges],
            "groups":[asdict(g) for g in groups],
            "excluded_parts":[asdict(k) for k in excluded],
            "boundary_touch_pairs":[[asdict(k) for k in pair] for pair in touches],
            "weak_overlap_pairs":[[asdict(k) for k in pair] for pair in weak],
            "counts":{"left_buildings":len(a),"right_buildings":len(b),"edges":len(edges),
                      "excluded_parts":len(excluded),"boundary_touches":len(touches),
                      "weak_overlaps":len(weak),"groups":dict(sorted(Counter(g.status for g in groups).items()))},
            "limitations":["uncalibrated spatial candidate rules","proximity is review only",
                           "no bridge-ID matching, parent resolution, identity assignment or geometry merge"]}
