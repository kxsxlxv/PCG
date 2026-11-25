"""Normalize verified 2-D OSM/Overture GeoJSON source records, without fusion.

GEOS validates topology; PROJ handles horizontal projection. No repairs,
clipping, inferred identity, measured-height claims, or vertical conversion.
"""
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re

from pyproj import Transformer, network
from pyproj.exceptions import ProjError
from shapely.geometry import shape, box
from shapely.ops import transform
from shapely.errors import GEOSException

from .source_snapshot import read_verified_geojson, SourceSnapshot, WorldFrame


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Observation:
    attribute: str
    value: str | float | int | None
    unit: str
    reference: str
    source_key: str
    raw_json: str
    status: str
    method: str = "source_reported"
    evidence_class: str = "unknown"
    uncertainty: float | None = None


@dataclass(frozen=True)
class SourceFeature:
    source_id: str
    snapshot_digest: str
    source_feature_id: str
    source_kind: str
    geometry_kind: str
    world_frame_digest: str
    local_geometry_wkb: str
    area_m2: float
    raw_json: str
    observations: tuple[Observation, ...]


@dataclass(frozen=True)
class RejectedFeature:
    source_id: str
    snapshot_digest: str
    source_feature_id: str | None
    reason: str
    raw_json: str


def _id(value):
    if type(value) is int:
        return str(value)
    if isinstance(value,str) and value.strip() and value.strip()==value:
        return value
    return None


def feature_id(f, kind):
    p=f.get("properties")
    p=p if isinstance(p,dict) else {}
    if kind=="osm":
        typ=p.get("@type")
        ident=_id(p.get("@id"))
        if typ in ("way","relation") and ident is not None:
            return typ+"/"+ident
        # osmium -u type_id; bare numeric IDs are ambiguous across types.
        ident=_id(f.get("id"))
        if ident and re.fullmatch(r"(?:way/|relation/|w|r)[0-9]+",ident):
            prefix="way" if ident.startswith("w") else "relation"
            return prefix+"/"+re.search(r"[0-9]+$",ident).group()
        return None
    return _id(p.get("id")) or _id(f.get("id"))


def _positive(value, *, metres=False, zero=False, integer=False, strings=False):
    if type(value) in (float,int):
        n=float(value)
    elif strings and isinstance(value,str):
        pattern=r"[+]?[0-9]+(?:[.,][0-9]+)?" + (r"(?:\s*m)?" if metres else "")
        if not re.fullmatch(pattern,value.strip(),re.I):
            return None
        n=float(re.sub(r"\s*m$","",value.strip(),flags=re.I).replace(",","."))
    else:
        return None
    if not math.isfinite(n) or n<0 or (not zero and n==0) or (integer and not n.is_integer()):
        return None
    return int(n) if integer else n


def observations(p,kind):
    # Do not conflate OSM height (top above ground) and Overture height
    # (bottom-to-top extent). Preserve raw source fields, even rejected values.
    osm=kind=="osm"
    fields=(
        ("height","height","m","top_agl" if osm else "vertical_extent",False,False),
        ("min_height","min_height","m","base_agl",True,False),
        ("building:levels" if osm else "num_floors","floors","count","above_ground",False,True),
        ("roof:height" if osm else "roof_height","roof_height","m","roof_vertical_extent",True,False),
    )
    out=[]
    for key,attr,unit,reference,zero,integer in fields:
        if key not in p:
            continue
        raw=p[key]
        value=_positive(raw,metres=unit=="m",zero=zero,integer=integer,strings=osm)
        status="missing" if raw is None else "candidate" if value is not None else "invalid"
        out.append(Observation(attr,value,unit,reference,key,canonical(raw),status))
    for key,attr,reference in (
            ("roof:shape" if osm else "roof_shape","roof_shape","source_enum"),
            ("building" if osm else "class","use","source_enum"),
            ("building:material" if osm else "facade_material","building_material" if osm else "facade_material","source_enum")):
        if key in p:
            raw=p[key]
            value=raw.strip() if isinstance(raw,str) and raw.strip().lower() not in ("","unknown","yes","no") else None
            out.append(Observation(attr,value,"none",reference,key,canonical(raw),
                                   "candidate" if value is not None else "missing" if raw is None else "unresolved"))
    return tuple(out)


def _coordinates(g):
    typ=g.get("type")
    if typ not in ("Polygon","MultiPolygon"):
        raise ValueError("nonpolygon_geometry")
    polys=[g.get("coordinates")] if typ=="Polygon" else g.get("coordinates")
    if not isinstance(polys,list) or not polys:
        raise ValueError("empty_geometry")
    for rings in polys:
        if not isinstance(rings,list) or not rings:
            raise ValueError("empty_ring_set")
        for ring in rings:
            if not isinstance(ring,list) or len(ring)<4 or ring[0]!=ring[-1]:
                raise ValueError("unclosed_or_short_ring")
            for xy in ring:
                if not isinstance(xy,list) or len(xy)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in xy):
                    raise ValueError("expected_finite_2d_coordinates")
                if not (-180<=xy[0]<=180 and -90<xy[1]<90):
                    raise ValueError("coordinate_outside_wgs84")
            if any(abs(a[0]-b[0])>180 for a,b in zip(ring,ring[1:])):
                raise ValueError("antimeridian_not_supported")


def normalize(snapshot:SourceSnapshot,payload,frame:WorldFrame,kind,bbox):
    if kind not in ("osm","overture_building","overture_part"):
        raise ValueError("explicit supported source kind required")
    if len(bbox)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bbox):
        raise ValueError("invalid bbox")
    w,s,e,n=bbox
    if not (-180<=w<e<=180 and -90<s<n<90):
        raise ValueError("invalid bbox")
    if network.is_network_enabled():
        raise ValueError("PROJ network must be disabled")
    data=read_verified_geojson(snapshot,payload)
    aoi=box(w,s,e,n)
    project=Transformer.from_crs("OGC:CRS84",frame.horizontal_crs,always_xy=True,allow_ballpark=False)
    def local(x,y):
        import numpy as np
        a,b=project.transform(x,y,errcheck=True)
        return np.asarray(a)-frame.origin[0],np.asarray(b)-frame.origin[1]
    ids=[feature_id(f,kind) for f in data["features"]]
    counts=Counter(i for i in ids if i is not None)
    accepted=[]; rejected=[]
    sd=snapshot.digest();fd=frame.digest()
    for f,fid in zip(data["features"],ids):
        raw=canonical(f)
        try:
            if fid is None:
                raise ValueError("missing_stable_source_id")
            if counts[fid]>1:
                raise ValueError("duplicate_source_id")
            p=f.get("properties")
            if not isinstance(p,dict):
                raise ValueError("invalid_properties")
            role="part" if kind=="overture_part" else "building"
            if kind=="osm":
                part=p.get("building:part")
                building=p.get("building")
                if isinstance(part,str) and part.strip().lower() not in ("","no","unknown"):
                    role="part"
                elif not isinstance(building,str) or building.strip().lower() in ("","no","unknown"):
                    raise ValueError("not_building")
            if p.get("is_underground") is True:
                raise ValueError("underground")
            g=f.get("geometry")
            if not isinstance(g,dict):
                raise ValueError("missing_geometry")
            _coordinates(g)
            geom=shape(g)
            if geom.is_empty or not geom.is_valid or geom.area<=0:
                raise ValueError("invalid_geometry")
            if not aoi.covers(geom.representative_point()):
                raise ValueError("outside_aoi_owner")
            metric=transform(local,geom)
            if metric.is_empty or not metric.is_valid or not math.isfinite(metric.area) or metric.area<=0:
                raise ValueError("invalid_projected_geometry")
            accepted.append(SourceFeature(snapshot.source_id,sd,fid,kind,role,fd,
                            metric.normalize().wkb_hex,float(metric.area),raw,observations(p,kind)))
        except (ValueError,GEOSException,ProjError,OverflowError) as exc:
            rejected.append(RejectedFeature(snapshot.source_id,sd,fid,str(exc),raw))
    accepted.sort(key=lambda f:f.source_feature_id)
    rejected.sort(key=lambda f:(f.source_feature_id or "",f.reason,f.raw_json))
    return tuple(accepted),tuple(rejected)


def report(accepted,rejected):
    return {"schema_version":1,"compiler_version":"0.1.0",
            "accepted":[asdict(x) for x in accepted],
            "rejected":[asdict(x) for x in rejected],
            "counts":{"input":len(accepted)+len(rejected),"accepted":len(accepted),
                      "rejected":len(rejected)},
            "reasons":dict(sorted(Counter(x.reason for x in rejected).items()))}
