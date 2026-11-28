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
from .volume_budget import Offer, select, detail_reserve
from shapely.geometry import Polygon

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


@dataclass(frozen=True)
class RoofVolume(RoofCore):
    width_m: float
    length_m: float
    height_m: float

@dataclass(frozen=True)
class VolumePlan(SectionPlan):
    reserved_triangles: int
    planned_triangles: int
    accepted_offers: tuple[str,...]
    volume_mode: bool=True


def baseline(f,rule,floors,top,*,roof='flat',eave=None,party_edges=()):
    # The legacy selector counts 20 triangles per anchor; full glazed modules
    # can cost 30. Limit its aperture selection conservatively, then give the
    # existing detail planner the full building cap. No anchors are deleted.
    v=plan_variant(f,rule.use,floors,top,archetype=rule.base,roof=roof,eave_m=eave,
                   party_edges=party_edges,max_triangles=1100)
    return replace(v,max_triangles=1800)


@dataclass(frozen=True)
class MeasuredProfile:
    # All values are dimensionless donor observations. The GIS floor height
    # supplies metric scale; calibrated values are bounded design transfers.
    spacing_over_floor: float
    width_over_floor: float
    height_over_floor: float
    sill_over_floor: float
    relief_over_short_side: float | None = None
    relief_phase: float | None = None
    relief_width_fraction: float | None = None
    roof_area_fraction: float | None = None
    roof_height_over_body: float | None = None

    def __post_init__(self):
        bounds={'spacing_over_floor':(.4,3.5),'width_over_floor':(.05,3.5),
                'height_over_floor':(.04,1.),'sill_over_floor':(0.,1.),
                'relief_over_short_side':(.005,.45),'relief_phase':(0.,1.),
                'relief_width_fraction':(.025,.55),'roof_area_fraction':(.01,.45),
                'roof_height_over_body':(.01,.30)}
        for k,(lo,hi) in bounds.items():
            v=getattr(self,k)
            if v is None and k.startswith(('relief_','roof_')):continue
            if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:
                raise ValueError('invalid measured profile: '+k)

def clamp(x,lo,hi):return min(hi,max(lo,x))

def build(f,rule_id,floors,top,*,roof='flat',eave=None,party_edges=(),volume=False,measured=None):
    rule=RULES[rule_id];old=baseline(f,rule,floors,top,roof=roof,eave=eave,party_edges=party_edges)
    h=old.eave_m;fh=h/floors;party=set(party_edges)
    if measured is not None:
        if not volume or rule.use!='apartments' or not isinstance(measured,MeasuredProfile):
            raise ValueError('measured profiles require apartment volume grammar')
        # Preserve credible piers and enough space for complete external
        # balcony modules. Do not equate roughness components with window IDs.
        spacing=clamp(fh*measured.spacing_over_floor,2.50,3.55)
        width=clamp(fh*measured.width_over_floor,1.10,min(1.72,spacing-.88))
        rule=replace(rule,spacing=spacing,width=width)
    if fh<2.55:raise ValueError('reference grammar requires adequate resolved floor height')
    dw=min(1.70,rule.spacing-.55);dh=min(2.20,fh-.30)
    sill=min(.85,fh*.27);wh=min(1.60 if rule.use!='office' else 1.92,fh-sill-.22)
    if measured is not None:
        sill=clamp(fh*measured.sill_over_floor,.68,.96)
        wh=clamp(fh*measured.height_over_floor,1.12,min(1.70,fh-sill-.24))
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
    # Rotation-invariant physical short side, not the axis-aligned GIS bbox.
    obb=list(f.polygon().minimum_rotated_rectangle.exterior.coords)
    short_side=min(math.dist(a,b) for a,b in zip(obb[:-1],obb[1:]) if math.dist(a,b)>1e-8)
    slots={};entry_candidates=[];projs=[];cores=[];facades=[];columns=[];envelopes={};wall_frames={};host=f.polygon()
    for r,ring in enumerate(rings):
        for i,(x,y) in enumerate(zip(ring,ring[1:]+ring[:1])):
            if (r,i) in party:continue
            fid=f'{f.building_id}:facade:{r}:{i}';length=math.dist(x,y)
            centers=window_offsets(length,profile.window_width,profile.spacing,profile.margin)
            count=len(centers);long=r==0 and length>=max(8.,.40*longest)
            wall_frames[r,i]=(x,y,length,centers)
            # Keep each section's core in a stable bay; partial final sections
            # stay coherent rather than filling the remainder with random bays.
            groups=[list(range(k,min(k+rule.section,count))) for k in range(0,count,rule.section)]
            core=[g[len(g)//2] for g in groups if len(g)>=3]
            proj=[g[0]+1 for g in groups if len(g)>=4] if long and floors>=3 and rule.projection else []
            if measured is not None and measured.relief_phase is not None and proj:
                # A source offset selects an interior bay in each section.
                # Preserve space for a complete balcony stack on its other side.
                proj=[g[1+min(len(g)-3,int(measured.relief_phase*(len(g)-2)))] for g in groups if len(g)>=4]
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
                depth=rule.projection
                if volume:
                    factor={'walkup_sections':.22,'long_slab':.30,'section_slab':.45,'tower_stacks':.60,'tall_bay_tower':.80,'office_frame':.22}[rule.id]
                    depth=min(2.40,max(depth,fh*factor*(.85+.30*(stable('volume-depth',f.building_id,i,b)%101)/100)))
                    group=next(g for g in groups if b in g)
                    if rule.id in ('section_slab','tower_stacks','tall_bay_tower') and b+1<len(centers) and b+1 in group:
                        U=centers[b+1]+half
                    if measured is not None and measured.relief_over_short_side is not None:
                        depth=clamp(short_side*measured.relief_over_short_side,.24,min(2.40,fh*.80))
                        wide=measured.relief_width_fraction is not None and measured.relief_width_fraction*len(group)>=1.45
                        U=centers[b+1]+half if wide and b+1 in group else c+half
                p=Projection(fid+f':projected-bay:{b}',fid,b,c-half,U,fh+.06,h-.12,depth)
                if volume:
                    tx,ty=(y[0]-x[0])/length,(y[1]-x[1])/length
                    q=Polygon([(x[0]+u*tx+d*ty,x[1]+u*ty-d*tx) for u,d in ((p.u,0),(p.U,0),(p.U,p.depth),(p.u,p.depth))])
                    if q.intersection(host).area>1e-8:continue
                    envelopes[p.id]=q
                projs.append(p)
            if r==0 and i==main and roof=='flat' and rule.roof_cores:
                for b in core:
                    k=fid+f':roof-core:{b}';depth=min(6.,max(2.5,longest*.08))
                    if volume:
                        width=min(4.6,max(2.55,rule.spacing*1.35));length_m=min(3.2,max(1.85,fh*.80))
                        height=min(2.85,max(1.55,fh*(.68 if floors>=10 else .53)))
                        if measured is not None and measured.roof_area_fraction is not None:
                            # Transfer an observed top-tier fraction into one
                            # small optional core per section, never the GIS shell.
                            area=host.area*measured.roof_area_fraction/max(1,len(core))
                            aspect=width/length_m
                            width=clamp(math.sqrt(area*aspect),2.25,4.60)
                            length_m=clamp(math.sqrt(area/aspect),1.65,3.20)
                        if measured is not None and measured.roof_height_over_body is not None:
                            height=clamp(h*measured.roof_height_over_body,1.45,2.85)
                        cores.append(RoofVolume(k,fid,centers[b],depth,width,length_m,height))
                    else:cores.append(RoofCore(k,fid,centers[b],depth))
            facades.append(dict(facade=fid,long=long,section_bays=rule.section,core_bays=core,projected_bays=proj,entry_bays=entries,centers=centers))
            for floor in range(floors):
                for b in range(count):
                    blank=(not long and r==0 and (rule.ends=='blind' or rule.ends=='central' and abs(b-(count-1)/2)>.6)) or (floor==0 and b in entries)
                    slots[r,i,floor,b]='blank' if blank else 'window'
            if long and rule.balconies:
                for g in groups:
                    b=g[-2] if len(g)>=4 else None
                    if measured is not None and b is not None:
                        # Keep a complete external stack outside all projected
                        # bays, rather than silently deleting its apertures.
                        blocked={k for k in g if any(p.facade==fid and p.u<centers[k]<p.U for p in projs)}
                        options=[k for k in g[1:-1] if k not in blocked and k not in core]
                        if options:b=max(options,key=lambda k:(abs(k-(g[0]+g[-1])/2),k))
                    if b is None or b in proj:continue
                    s=tuple((r,i,fl,b) for fl in range(1,floors) if slots[r,i,fl,b]=='window')
                    if s:columns.append((stable('ref-column',f.building_id,rule.id,i,b),s))
    # Full stacks only; 30 tris per glazed balcony is a conservative bound.
    # Reserve 650 triangles for parapets, projection strips, entries and roof.
    reserve=650;selected=();planned=0;used=0
    shell=len(old.mass.shell.triangles())
    if volume:
        # Keep independent outward volumes nonoverlapping before spending on
        # complete balcony stacks. Same-building footprint remains immutable.
        accepted=[]
        for p in sorted(projs,key=lambda p:(-p.depth,stable('volume',p.id),p.id)):
            if not any(envelopes[p.id].intersection(envelopes[q.id]).area>1e-8 for q in accepted):accepted.append(p)
        projs=accepted
        # Reserve the conservative widened balcony envelope, not just its door.
        safe=[]
        for tie,col in columns:
            r,i,_,b=col[0];x,y,L,centers=wall_frames[r,i]
            tx,ty=(y[0]-x[0])/L,(y[1]-x[1])/L;u=centers[b]
            width=min(2.10,profile.spacing-.20)+.40
            q=Polygon([(x[0]+a*tx+d*ty,x[1]+a*ty-d*tx) for a,d in ((u-width/2,0),(u+width/2,0),(u+width/2,1.30),(u-width/2,1.30))])
            if not any(q.intersection(envelopes[p.id]).area>1e-8 for p in projs):safe.append((tie,col))
        offers=[Offer(p.id,10,3,stable('offer',p.id)) for p in projs]
        offers.extend(Offer(q.id,24,2,stable('offer',q.id)) for q in cores)
        def colid(col):return f'{f.building_id}:stack:{col[0][0]}:{col[0][1]}:{col[0][3]}'
        offers.extend(Offer(colid(col),30*len(col),1,tie) for tie,col in safe)
        reserve=detail_reserve(rings,len(entry_candidates),roof)
        selected,planned=select(offers,max(0,1800-shell-reserve))
        chosen=set(selected);projs=[p for p in projs if p.id in chosen];cores=[q for q in cores if q.id in chosen]
        for _,col in safe:
            if colid(col) in chosen:
                for key in col:slots[key]='balcony'
                used+=len(col)
    else:
        capacity=max(0,(1800-reserve-shell)//30)
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
    assert count<=used and count*30+shell<=max(shell,1800-reserve)
    v=replace(old,archetype=a,selection='reference_rule',plan=plan,requested_balcony_slots=sum(len(x) for _,x in columns),budget_omitted_slots=sum(len(x) for _,x in columns)-count)
    if volume:
        assert count*30+10*len(projections)+24*len(cores)<=planned
        return replace(v,selection='volume_rule'),VolumePlan(rule,tuple(entries),tuple(projections),tuple(cores),tuple(facades),reserve,planned,selected)
    return v,SectionPlan(rule,tuple(entries),tuple(projections),tuple(cores),tuple(facades))
