"""Opt-in, reference-informed architectural section grammar.
Authored generic rules, not a catalog of identified real-world series. The
canonical GIS shell is immutable. Additional facade and roof intents live in
an explicit research sidecar; no legacy PCGF v1 contract is changed.
"""
from dataclasses import dataclass, replace, asdict
import math
from .archetypes import plan_variant, stable
from .architecture import plan_facades
from ._grammar_bays import window_offsets

@dataclass(frozen=True)
class Rule:
    id: str
    base: str
    use: str
    pitch: float
    spacing: float
    width: float
    section: int
    ends: str
    projection: float
    balconies: bool
    repeat_entries: bool
    roof_cores: bool
    composition: str

# Measured reference floor periods motivated the pitch priors; all dimensions,
# floor counts, section widths and construction choices here are authored.
RULES={r.id:r for r in (
    Rule('walkup_sections','panel_slab','apartments',2.90,2.85,1.30,6,'central',.24,True,True,True,'blind_end'),
    Rule('long_slab','panel_slab','apartments',2.90,2.95,1.38,8,'blind',.40,True,True,True,'blind_end'),
    Rule('section_slab','panel_slab','apartments',2.92,3.10,1.46,9,'central',.52,True,True,True,'sections'),
    Rule('tower_stacks','caucasus_tower','apartments',3.10,3.05,1.58,7,'central',.60,True,True,True,'sections'),
    Rule('tall_bay_tower','caucasus_tower','apartments',3.10,3.15,1.55,7,'central',.85,False,True,True,'blind_end'),
    Rule('pitched_residential','caucasus_house','house',3.05,2.85,1.25,4,'all',0.,False,False,False,'paired'),
    Rule('office_frame','modern_ribbon','office',3.35,3.25,2.20,5,'central',.27,False,True,True,'ribbon'),
    Rule('service_hall','civic_block','office',4.30,4.50,2.40,5,'blind',0.,False,True,False,'sections'),
)}

@dataclass(frozen=True)
class Entry:
    id: str
    facade: str
    bay: int
    rect: tuple[float,float,float,float]

@dataclass(frozen=True)
class Projection:
    id: str
    facade: str
    bay: int
    u: float
    U: float
    z: float
    Z: float
    depth: float

@dataclass(frozen=True)
class RoofCore:
    id: str
    facade: str
    u: float
    depth: float

@dataclass(frozen=True)
class SectionPlan:
    rule: Rule
    entries: tuple[Entry,...]
    projections: tuple[Projection,...]
    roof_cores: tuple[RoofCore,...]
    facades: tuple[dict,...]
    def canonical(self):return asdict(self)


def baseline(f,rule,floors,top,*,roof='flat',eave=None,party_edges=()):
    # The legacy selector counts 20 triangles per anchor; full glazed modules
    # can cost 30. Limit its aperture selection conservatively, then give the
    # existing detail planner the full building cap. No anchors are deleted.
    v=plan_variant(f,rule.use,floors,top,archetype=rule.base,roof=roof,eave_m=eave,
                   party_edges=party_edges,max_triangles=1100)
    return replace(v,max_triangles=1800)


def build(f,rule_id,floors,top,*,roof='flat',eave=None,party_edges=()):
    rule=RULES[rule_id];old=baseline(f,rule,floors,top,roof=roof,eave=eave,party_edges=party_edges)
    h=old.eave_m;fh=h/floors;party=set(party_edges)
    if fh<2.55:raise ValueError('reference grammar requires adequate resolved floor height')
    dw=min(1.70,rule.spacing-.55);dh=min(2.20,fh-.30)
    sill=min(.85,fh*.27);wh=min(1.60 if rule.use!='office' else 1.92,fh-sill-.22)
    if rule.id=='service_hall':
        dw=min(3.2,rule.spacing-.55);dh=min(3.2,fh-.30);sill=min(2.65,fh-.95);wh=min(.8,fh-sill-.2)
    dims=(rule.width,wh,sill,rule.spacing,max(.65,(rule.spacing-rule.width)/2),dw,dh,.14)
    a=replace(old.archetype,dimensions=dims,composition=rule.composition,
              enclosure='mixed' if rule.balconies else old.archetype.enclosure)
    if rule.id=='service_hall':a=replace(a,wall='concrete_panel',pitched_material='metal_roof_red')
    profile=a.profile();rings=(f.exterior,*f.holes)
    exterior=[(math.dist(x,y),i) for i,(x,y) in enumerate(zip(f.exterior,f.exterior[1:]+f.exterior[:1])) if (0,i) not in party]
    main=max(exterior,key=lambda q:(q[0],-q[1]))[1] if exterior else None
    longest=max((x[0] for x in exterior),default=1.)
    slots={};entry_candidates=[];projs=[];cores=[];facades=[];columns=[]
    for r,ring in enumerate(rings):
        for i,(x,y) in enumerate(zip(ring,ring[1:]+ring[:1])):
            if (r,i) in party:continue
            fid=f'{f.building_id}:facade:{r}:{i}';length=math.dist(x,y)
            centers=window_offsets(length,profile.window_width,profile.spacing,profile.margin)
            count=len(centers);long=r==0 and length>=max(8.,.40*longest)
            # Keep each section's core in a stable bay; partial final sections
            # stay coherent rather than filling the remainder with random bays.
            groups=[list(range(k,min(k+rule.section,count))) for k in range(0,count,rule.section)]
            core=[g[len(g)//2] for g in groups if len(g)>=3]
            proj=[g[0]+1 for g in groups if len(g)>=4] if long and floors>=3 and rule.projection else []
            entries=[]
            if r==0 and i==main and rule.repeat_entries:
                for b in core:
                    c=centers[b];u,U=c-dw/2,c+dw/2
                    # The canonical primary door is centred; supplemental doors
                    # never compete with it, even if the primary is suppressed.
                    if abs(c-length/2)<dw+.12:continue
                    if u>.08 and U<length-.08:
                        rect=(u,.05,U,.05+dh);entries.append(b);entry_candidates.append(Entry(fid+f':section-entry:{b}',fid,b,rect))
            for b in proj:
                c=centers[b];half=min(rule.width/2+.18,rule.spacing/2-.09)
                # Wider two-bay shafts are reserved for the tall tower family.
                U=centers[b+1]+half if rule.id=='tall_bay_tower' else c+half
                projs.append(Projection(fid+f':projected-bay:{b}',fid,b,c-half,U,fh+.06,h-.12,rule.projection))
            if r==0 and i==main and roof=='flat' and rule.roof_cores:
                for b in core:cores.append(RoofCore(fid+f':roof-core:{b}',fid,centers[b],min(6.,max(2.5,longest*.08))))
            facades.append(dict(facade=fid,long=long,section_bays=rule.section,core_bays=core,projected_bays=proj,entry_bays=entries,centers=centers))
            for floor in range(floors):
                for b in range(count):
                    blank=(not long and r==0 and (rule.ends=='blind' or rule.ends=='central' and abs(b-(count-1)/2)>.6)) or (floor==0 and b in entries)
                    slots[r,i,floor,b]='blank' if blank else 'window'
            if long and rule.balconies:
                for g in groups:
                    b=g[-2] if len(g)>=4 else None
                    if b is None or b in proj:continue
                    s=tuple((r,i,fl,b) for fl in range(1,floors) if slots[r,i,fl,b]=='window')
                    if s:columns.append((stable('ref-column',f.building_id,rule.id,i,b),s))
    # Full stacks only; 30 tris per glazed balcony is a conservative bound.
    # Reserve 650 triangles for parapets, projection strips, entries and roof.
    capacity=max(0,(1800-650-len(old.mass.shell.triangles()))//30);used=0
    for _,col in sorted(columns):
        if used+len(col)<=capacity:
            for key in col:slots[key]='balcony'
            used+=len(col)
    plan=plan_facades(f,h,floors,profile,party_edges=party_edges,
                      entrance_edge=None if main is None else (0,main),
                      slot_policy=lambda r,i,fl,b,n,l:slots.get((r,i,fl,b),'blank'))
    by={w.id:w for w in plan.facades};entries=[]
    for e in entry_candidates:
        w=by[e.facade];u,z,U,Z=e.rect
        if not any(min(U,o.u+o.width+.035)>max(u,o.u-.035) and min(Z,o.z+o.height+.035)>max(z,o.z-.035) for o in w.openings):entries.append(e)
    projections=[]
    for p in projs:
        w=by[p.facade];assert p.u>0 and p.U<w.length and 0<p.z<p.Z<=h
        # All windows in a projected bay are mapped to its actual front face;
        # no balcony door or neighboring aperture is hidden by the projection.
        targets=[o for o in w.openings if min(p.U,o.u+o.width)>max(p.u,o.u) and min(p.Z,o.z+o.height)>max(p.z,o.z)]
        if targets and all(o.role=='window' and p.u<o.u and o.u+o.width<p.U and p.z<o.z and o.z+o.height<p.Z for o in targets):projections.append(p)
    assert plan.footprint_digest==old.plan.footprint_digest==f.digest()
    count=sum(o.role=='balcony_door' for w in plan.facades for o in w.openings)
    assert count<=used and count*30+len(old.mass.shell.triangles())<=1800-650
    v=replace(old,archetype=a,selection='reference_rule',plan=plan,requested_balcony_slots=sum(len(x) for _,x in columns),budget_omitted_slots=sum(len(x) for _,x in columns)-count)
    return v,SectionPlan(rule,tuple(entries),tuple(projections),tuple(cores),tuple(facades))
