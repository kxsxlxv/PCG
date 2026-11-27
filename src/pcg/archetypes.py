"""Opt-in architectural grammar v1. Synthetic regional priors, not GIS observations.
Reuses audited bay packing, canonical aperture/anchor exclusion and L2 massing.
The three legacy profile IDs and packed regular-cell ABI stay unchanged.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from .architecture import Profile, PROFILES, ArchitecturePlan, plan_facades
from .contracts import number
from ._grammar_bays import window_offsets
from .massing import Massing, build_massing

REVISION = 1
MAX_BALCONY_TRIANGLES = 20  # closed slab + 3 guards + optional top cover
REGIONS = ("generic", "baltic", "nordic", "central_europe", "mediterranean", "caucasus")


def stable(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, ensure_ascii=False,
        separators=(",", ":")).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Archetype:
    id: str
    use: str
    regions: tuple[str, ...]
    levels: tuple[int, int]
    # width, height, sill, pitch, margin, entry width, entry height, recess
    dimensions: tuple[float, ...]
    roof: str
    rise_ratio: float
    wall: str
    pitched_material: str
    composition: str
    balcony: str
    enclosure: str
    accent: tuple[float, float, float]

    def profile(self):
        return Profile(self.use, *self.dimensions)


# Declarative presets, not learned weights or identified real building series.
# Roof preference only applies when the source has not resolved a roof.
# Accent is a shared linear color multiplier, not a unique texture allocation.
ROWS = (
    ("baltic_timber", "house", ("baltic", "nordic"), (1, 3), (1.05,1.42,.78,2.50,.65,1.0,2.05,.14), "gable", .29, "plaster_cream", "metal_roof_red", "paired", "none", "open", (.76,.86,.72)),
    ("nordic_cottage", "house", ("nordic", "baltic"), (1, 3), (1.55,1.55,.65,3.10,.75,1.10,2.10,.12), "gable", .34, "plaster_cream", "metal_roof_red", "grid", "none", "open", (.75,.56,.45)),
    ("central_townhouse", "house", ("central_europe", "baltic"), (1, 4), (.95,1.75,.65,2.25,.65,1.15,2.20,.18), "gable", .27, "red_brick", "ceramic_roof", "paired", "none", "open", (.97,.86,.75)),
    ("mediterranean_villa", "house", ("mediterranean", "caucasus"), (1, 3), (1.35,1.65,.65,3.40,.75,1.40,2.15,.20), "hip", .18, "plaster_white", "ceramic_roof", "corner", "none", "open", (1.05,.94,.78)),
    ("caucasus_house", "house", ("caucasus",), (1, 4), (1.45,1.45,.78,2.85,.85,1.20,2.10,.15), "hip", .22, "plaster_cream", "metal_roof_red", "sections", "none", "open", (.89,.91,1.02)),
    ("panel_slab", "apartments", ("baltic", "central_europe", "caucasus"), (4, 16), (1.40,1.45,.82,2.80,.75,1.25,2.15,.10), "flat", 0., "concrete_panel", "bitumen_roof", "blind_end", "columns", "glazed", (.87,.88,.90)),
    ("brick_walkup", "apartments", ("baltic", "central_europe"), (2, 6), (1.15,1.55,.75,2.45,.75,1.20,2.15,.17), "gable", .20, "red_brick", "ceramic_roof", "grid", "alternate", "mixed", (1.04,.94,.85)),
    ("perimeter_block", "apartments", ("central_europe", "baltic"), (3, 9), (1.10,1.72,.68,2.50,.72,1.45,2.25,.19), "hip", .16, "plaster_cream", "ceramic_roof", "sections", "ends", "open", (1.00,.91,.79)),
    ("mediterranean_balcony", "apartments", ("mediterranean",), (3, 16), (1.70,1.55,.70,2.90,.80,1.50,2.20,.15), "flat", 0., "plaster_white", "bitumen_roof", "grid", "paired", "open", (1.02,.93,.81)),
    ("caucasus_tower", "apartments", ("caucasus",), (5, 40), (1.60,1.55,.75,3.05,.70,1.55,2.25,.14), "flat", 0., "concrete_panel", "bitumen_roof", "corner", "ends", "mixed", (.90,.95,1.06)),
    ("modern_gallery", "apartments", ("nordic", "generic", "central_europe"), (2, 40), (1.90,1.65,.60,3.20,.80,1.55,2.25,.14), "flat", 0., "plaster_white", "bitumen_roof", "paired", "alternate", "open", (.83,.95,1.03)),
    ("masonry_office", "office", ("baltic", "central_europe"), (1, 12), (1.20,1.90,.65,2.55,.80,1.80,2.35,.20), "hip", .14, "red_brick", "ceramic_roof", "sections", "none", "open", (.98,.90,.80)),
    ("modern_ribbon", "office", ("nordic", "generic", "caucasus"), (2, 60), (2.60,1.55,.90,3.00,.60,2.10,2.35,.10), "flat", 0., "concrete_panel", "bitumen_roof", "ribbon", "none", "open", (.88,.94,1.04)),
    ("glass_grid", "office", ("generic", "mediterranean", "caucasus"), (2, 100), (2.20,2.35,.30,2.55,.65,2.20,2.35,.08), "flat", 0., "plaster_white", "bitumen_roof", "grid", "none", "open", (.71,.85,.98)),
    ("civic_block", "office", ("central_europe", "baltic", "mediterranean"), (1, 8), (1.35,2.00,.65,3.20,.95,2.20,2.45,.21), "gable", .17, "stone_plinth", "ceramic_roof", "paired", "none", "open", (1.02,.98,.89)),
)
ARCHETYPES = {row[0]: Archetype(*row) for row in ROWS}


def select_archetype(building_id, use, floors, region="generic", *, override=None):
    if not isinstance(building_id, str) or not building_id.strip():
        raise ValueError("building_id required")
    if use not in PROFILES or region not in REGIONS or type(floors) is not int or not 1 <= floors <= 200:
        raise ValueError("invalid archetype selection contract")
    eligible = [a for a in ARCHETYPES.values() if a.use == use and a.levels[0] <= floors <= a.levels[1]]
    if override is not None:
        if override not in {a.id for a in eligible}:
            raise ValueError("incompatible explicit archetype")
        return ARCHETYPES[override]
    if not eligible:
        raise ValueError("no archetype compatible with resolved use and floors")
    # Stable sorted candidates. Nonmatching styles retain a small weight so
    # regional priors are not hard exclusions; generic is deliberately neutral.
    candidates = [(a, 4 if region != "generic" and region in a.regions else 1)
                  for a in sorted(eligible, key=lambda a: a.id)]
    index = stable("archetype-v1", building_id, use, region, floors) % sum(w for _, w in candidates)
    for a, weight in candidates:
        if index < weight:
            return a
        index -= weight
    raise AssertionError("selection index")


def _massing(f, a, top, floors, roof, eave):
    observed_roof, observed_eave = roof is not None, eave is not None
    if observed_roof and roof not in ("flat", "gable", "hip"):
        raise ValueError("unsupported observed roof")
    p = a.profile()
    if observed_eave:
        h = number(eave, "eave_m")
        if not 0 < h <= top:
            raise ValueError("observed eave outside source height")
        requested = roof if observed_roof else ("flat" if h == top else a.roof if a.roof != "flat" else "hip")
        rise = top - h
        if (requested == "flat") != (rise == 0):
            raise ValueError("observed roof and eave conflict")
    else:
        requested = roof if observed_roof else a.roof
        # Rotated buildings use minimum rotated rectangle dimensions, not AABB.
        ring = list(f.polygon().minimum_rotated_rectangle.exterior.coords)
        short = min(math.dist(x, y) for x, y in zip(ring, ring[1:]))
        min_floor = max(2.55, p.sill+p.window_height+.10, p.door_height+.25)
        rise = 0. if requested == "flat" else min(short*(a.rise_ratio or .18), top*.28, top-floors*min_floor)
        if requested != "flat" and rise < .45:
            if observed_roof:
                raise ValueError("observed pitched roof cannot fit resolved height/floors")
            return Massing(build_massing(f, top).shell, requested, "flat", "insufficient_height_for_inferred_pitch"), top
        h = top - rise
    mass = build_massing(f, h, requested, rise)
    if mass.fallback_reason:
        if observed_roof or observed_eave:
            raise ValueError("observed roof/eave requires review: " + mass.fallback_reason)
        # Inferred shape may fall back but source TOP height must never shrink.
        return Massing(build_massing(f, top).shell, requested, "flat", mass.fallback_reason), top
    return mass, h


def _blank(a, r, floor, bay, count, long):
    if r:
        return False
    if a.composition == "blind_end" and not long:
        return True
    if count < 3:
        return False
    if a.composition == "paired" and count >= 5:
        return bay % 3 == 2 and bay != count-1
    if a.composition == "sections" and count >= 6:
        return bay % 4 == 3 and bay != count-1
    if a.composition == "corner" and count >= 4:
        return bay in (0, count-1) and not long
    return False


def _balcony(a, floor, bay, count, long, phase):
    if not long or floor == 0 or count == 0:
        return False
    if a.balcony == "columns":
        return (bay + phase) % 3 == 0
    if a.balcony == "alternate":
        return (bay + phase) % 3 == 0 and floor % 2 == 1
    if a.balcony == "paired":
        return (bay + phase) % 4 in (0, 1)
    if a.balcony == "ends":
        return bay in (0, count-1)
    return False


@dataclass(frozen=True)
class Variant:
    archetype: Archetype
    region: str
    selection: str
    roof_source: str
    eave_source: str
    top_m: float
    eave_m: float
    plan: ArchitecturePlan
    mass: Massing
    requested_balcony_slots: int
    budget_omitted_slots: int
    max_triangles: int

    def glazed(self, facade_id, opening_id):
        """Column-consistent finish; stable regardless of build/LOD order."""
        if self.archetype.enclosure != "mixed":
            return self.archetype.enclosure == "glazed"
        bay = opening_id.rsplit(":", 1)[-1]
        return stable("enclosure-v1", facade_id, bay) % 3 != 0

    def decision(self):
        return dict(revision=REVISION, archetype=asdict(self.archetype), region=self.region,
            selection=self.selection, roof_source=self.roof_source, eave_source=self.eave_source,
            top_m=self.top_m, eave_m=self.eave_m, requested_roof=self.mass.requested_roof,
            actual_roof=self.mass.actual_roof, fallback_reason=self.mass.fallback_reason,
            requested_balcony_slots=self.requested_balcony_slots, budget_omitted_slots=self.budget_omitted_slots,
            max_triangles=self.max_triangles, plan_digest=self.plan.digest())


def plan_variant(f, use, floors, top_m, *, region="generic", archetype=None,
                 roof=None, eave_m=None, party_edges=(), entrance_edge=None, max_triangles=1800):
    """Do not mutate GIS footprint, floor count, top or explicit roof/eave.
    Budget applies to host shell plus complete external balcony modules, not
    the later full-detail material/roof-equipment pipeline. Whole columns are
    admitted before aperture planning. Canonical fit can still reject a slot.
    """
    top = number(top_m, "top_m")
    if top <= 0 or type(max_triangles) is not int or max_triangles <= 0:
        raise ValueError("invalid height or triangle budget")
    a = select_archetype(f.building_id, use, floors, region, override=archetype)
    mass, h = _massing(f, a, top, floors, roof, eave_m)
    if not math.isclose(max(v[2] for v in mass.shell.vertices), top, rel_tol=0, abs_tol=1e-8):
        raise ValueError("source top height changed")
    n = len(mass.shell.triangles())
    if n > max_triangles:
        raise ValueError("host shell exceeds triangle budget; requires review")
    capacity = (max_triangles-n)//MAX_BALCONY_TRIANGLES
    profile = a.profile()
    party = tuple(party_edges)
    longest = max(math.dist(x,y) for x,y in zip(f.exterior, f.exterior[1:]+f.exterior[:1]))
    phase = stable("columns-v1", f.building_id, a.id) % 12
    columns = []
    for i, (x,y) in enumerate(zip(f.exterior, f.exterior[1:]+f.exterior[:1])):
        length = math.dist(x,y)
        long = length >= .78*longest
        centers = window_offsets(length, profile.window_width, profile.spacing, profile.margin)
        if (0,i) in party:
            continue
        for bay in range(len(centers)):
            slots = tuple((0,i,floor,bay) for floor in range(1,floors)
                if not _blank(a,0,floor,bay,len(centers),long)
                and _balcony(a,floor,bay,len(centers),long,phase))
            if slots:
                columns.append((stable("column-rank-v1", f.building_id, a.id, i, bay),i,bay,slots))
    chosen = set()
    requested = sum(len(slots) for _,_,_,slots in columns)
    for _,_,_,slots in sorted(columns):
        if len(chosen)+len(slots) <= capacity:
            chosen.update(slots)
    def policy(r,i,floor,bay,count,length):
        if _blank(a,r,floor,bay,count,length >= .78*longest):
            return "blank"
        return "balcony" if (r,i,floor,bay) in chosen else "window"
    plan = plan_facades(f,h,floors,profile,party_edges=party,entrance_edge=entrance_edge,slot_policy=policy)
    return Variant(a,region,"explicit" if archetype is not None else "stable_regional_prior",
        "observed" if roof is not None else "inferred", "observed" if eave_m is not None else "inferred",
        top,h,plan,mass,requested,requested-len(chosen),max_triangles)
