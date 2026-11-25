"""Conservative attribute resolution after explicit source identity adjudication.

No inferred identity from spatial scores, no geometry union, no terrain datum.
A caller supplies persistent internal IDs and explicitly chooses geometry.
"""
from dataclasses import asdict, dataclass
import math
from .geo_match import SourceKey, build_graph, key
from .geo_normalize import digest

ROOFS={"flat":"flat","gable":"gable","gabled":"gable","hip":"hip","hipped":"hip"}
USES={"house":"house","detached":"house","apartments":"apartments","office":"office"}
POLICY_REVISION=1


@dataclass(frozen=True)
class Approval:
    group_id: str
    building_id: str
    geometry_source: SourceKey
    reason: str


def _candidate(record, observation, value, *, assumption=None):
    return {"source":asdict(key(record)),"snapshot_digest":record.snapshot_digest,
            "source_key":observation.source_key,"reference":observation.reference,
            "value":value,"evidence_class":observation.evidence_class,
            "assumption":assumption}


def _select(candidates):
    """No provider ranking or averaging: only exact agreement, or abstain."""
    if not candidates:
        return {"status":"missing","value":None,"candidates":[]}
    values=[c["value"] for c in candidates]
    if any(v!=values[0] for v in values[1:]):
        return {"status":"conflict","value":None,"candidates":candidates}
    return {"status":"agreed" if len(values)>1 else "single_candidate",
            "value":values[0],"candidates":candidates}


def resolve_fields(records):
    fields={k:[] for k in ("base_agl_m","top_agl_m","roof_rise_m","floors","roof","profile")}
    notes=[];unusable=[]
    for record in sorted(records,key=key):
        if record.geometry_kind!="building":
            raise ValueError("building-level resolution only")
        obs={}
        for o in record.observations:
            if o.attribute in obs:
                raise ValueError("duplicate observation attribute")
            obs[o.attribute]=o
        def valid(name,reference,unit):
            o=obs.get(name)
            if o is None:
                return None
            if o.status!="candidate" or o.reference!=reference or o.unit!=unit:
                unusable.append({"source":asdict(key(record)),"attribute":name,"reason":"not_usable","observation":asdict(o)})
                return None
            return o
        def metric(name,ref,zero=False):
            o=valid(name,ref,"m")
            if o is None:return None
            if type(o.value) not in (int,float) or not math.isfinite(o.value) or o.value<0 or (not zero and o.value==0):
                raise ValueError("invalid normalized metric observation")
            return o
        base=metric("min_height","base_agl",True)
        base_missing="min_height" not in obs or obs["min_height"].status=="missing"
        if base is not None:
            fields["base_agl_m"].append(_candidate(record,base,base.value))
        elif base_missing:
            notes.append({"source":asdict(key(record)),"reason":"base_absent_ground_attached_assumption"})
            fields["base_agl_m"].append({"source":asdict(key(record)),"snapshot_digest":record.snapshot_digest,
                "source_key":None,"reference":"base_agl","value":0.,"evidence_class":"inferred",
                "assumption":"ordinary_building_ground_attached"})
        else:
            notes.append({"source":asdict(key(record)),"reason":"invalid_base_blocks_geometry"})
        ref="top_agl" if record.source_kind=="osm" else "vertical_extent"
        h=metric("height",ref)
        if h is not None:
            if ref=="top_agl":
                fields["top_agl_m"].append(_candidate(record,h,h.value))
            elif base is not None or base_missing:
                fields["top_agl_m"].append(_candidate(record,h,h.value+(base.value if base else 0.),
                    assumption="extent_plus_explicit_base" if base else "extent_plus_assumed_zero_base"))
            else:
                notes.append({"source":asdict(key(record)),"reason":"extent_without_valid_base"})
        rise=metric("roof_height","roof_vertical_extent",True)
        if rise is not None:fields["roof_rise_m"].append(_candidate(record,rise,rise.value))
        floors=valid("floors","above_ground","count")
        if floors is not None:
            if type(floors.value) is not int or not 1<=floors.value<=200:
                unusable.append({"source":asdict(key(record)),"attribute":"floors","reason":"outside_compiler_floor_range","observation":asdict(floors)})
            else:fields["floors"].append(_candidate(record,floors,floors.value))
        for attr,target,table in (("roof_shape","roof",ROOFS),("use","profile",USES)):
            o=valid(attr,"source_enum","none")
            if o is None:continue
            mapped=table.get(o.value.lower()) if isinstance(o.value,str) else None
            if mapped is None:
                unusable.append({"source":asdict(key(record)),"attribute":attr,"reason":"unsupported_or_generic_enum","observation":asdict(o)})
            else:fields[target].append(_candidate(record,o,mapped))
    decisions={name:_select(c) for name,c in fields.items()}
    vals={name:d["value"] for name,d in decisions.items()}
    blockers=[name+":"+d["status"] for name,d in decisions.items() if d["status"]=="conflict"]
    if any(n["reason"]=="invalid_base_blocks_geometry" for n in notes):
        blockers.append("invalid_base")
    for name in ("base_agl_m","top_agl_m","floors","roof","profile"):
        if vals[name] is None:blockers.append(name+":unresolved")
    if vals["roof"]=="flat" and decisions["roof_rise_m"]["status"]=="missing":
        vals["roof_rise_m"]=0.
        decisions["roof_rise_m"]={"status":"derived_flat_roof","value":0.,"candidates":[]}
    if vals["roof_rise_m"] is None:blockers.append("roof_rise_m:unresolved")
    if vals["roof"]=="flat" and vals["roof_rise_m"] not in (None,0):
        blockers.append("flat_roof_nonzero_rise")
    if vals["roof"] in ("gable","hip") and vals["roof_rise_m"]==0:
        blockers.append("pitched_roof_zero_rise")
    base,top,rise=(vals[k] for k in ("base_agl_m","top_agl_m","roof_rise_m"))
    eave=None
    if None not in (base,top,rise):
        eave=top-rise
        if eave<=base:blockers.append("eave_not_above_base")
        elif vals["floors"] is not None and not 2.0<=(eave-base)/vals["floors"]<=6.0:
            blockers.append("floor_height_outside_synthetic_policy")
    if base not in (None,0):
        blockers.append("elevated_building_not_supported")
    return {"policy_revision":POLICY_REVISION,"decisions":decisions,"values":{**vals,"eave_agl_m":eave},
            "notes":notes,"unusable":unusable,"blockers":sorted(set(blockers)),
            "status":"review_required" if blockers else "candidate_ready"}


def adjudicate(left,right,approvals,*,expected_graph_digest):
    left,right=tuple(left),tuple(right)
    graph=build_graph(left,right)
    if digest(graph)!=expected_graph_digest:
        raise ValueError("stale graph approval")
    groups={g["group_id"]:g for g in graph["groups"]}
    lookup={key(f):f for f in left+right}
    seen=set();internal_ids=set();resolved=[]
    for a in sorted(approvals,key=lambda a:a.building_id):
        if not isinstance(a,Approval) or not isinstance(a.building_id,str) or not a.building_id.strip() or not isinstance(a.reason,str) or not a.reason.strip():
            raise ValueError("explicit internal identity and reason required")
        if a.group_id in seen or a.building_id in internal_ids:
            raise ValueError("duplicate approval or internal ID")
        g=groups.get(a.group_id)
        if g is None or g["status"] not in ("isolated_high_candidate","unmatched_left","unmatched_right"):
            raise ValueError("group requires separate conflict adjudication")
        members=tuple(sorted(SourceKey(**k) for k in (*g["left"],*g["right"])))
        if a.geometry_source not in members:
            raise ValueError("geometry source not in approved group")
        records=tuple(lookup[k] for k in members)
        geometry=lookup[a.geometry_source]
        resolution=resolve_fields(records)
        resolved.append({"building_id":a.building_id,"group_id":a.group_id,"approval_reason":a.reason,
            "members":[{"key":asdict(key(f)),"snapshot_digest":f.snapshot_digest} for f in records],
            "geometry_source":asdict(a.geometry_source),"world_frame_digest":geometry.world_frame_digest,
            "local_geometry_wkb":geometry.local_geometry_wkb,"area_m2":geometry.area_m2,
            "resolution":resolution})
        seen.add(a.group_id);internal_ids.add(a.building_id)
    return {"schema_version":1,"compiler_version":"0.1.0","graph_digest":expected_graph_digest,
            "resolved":resolved,"unapproved_groups":[g for g in graph["groups"] if g["group_id"] not in seen],
            "excluded_parts":graph["excluded_parts"],
            "limitations":["explicit fixture/caller approvals only","no persistent registry storage or bridge-ID validation",
                           "no automatic split/merge, calibrated uncertainty, terrain or mesh generation"]}
