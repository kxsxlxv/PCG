"""Canonical local metric polygon contract. Invalid input is rejected, never repaired silently."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from .contracts import number


def _ring(raw, ccw):
    if not isinstance(raw, (list, tuple)):
        raise ValueError("ring must be an array")
    pts = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("expected XY coordinate")
        pts.append(tuple(number(x, "coordinate") or 0.0 for x in p))
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3 or len(set(pts)) != len(pts):
        raise ValueError("ring requires at least three distinct vertices without repetitions")
    poly = Polygon(pts)
    if not poly.is_valid or not math.isfinite(poly.area) or poly.area <= 0:
        raise ValueError("invalid ring: " + explain_validity(poly))
    if poly.exterior.is_ccw != ccw:
        pts.reverse()
    start = min(range(len(pts)), key=pts.__getitem__)
    return tuple(pts[start:] + pts[:start])


@dataclass(frozen=True)
class Footprint:
    building_id: str
    exterior: tuple
    holes: tuple = ()
    schema_version: int = 1
    units: str = "m"
    frame: str = "local_z_up"

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported schema version")
        if self.units != "m" or self.frame != "local_z_up":
            raise ValueError("expected local metric Z-up frame")
        if not isinstance(self.building_id, str) or not self.building_id.strip():
            raise ValueError("building_id required")
        if not isinstance(self.holes, (list, tuple)):
            raise ValueError("holes must be an array")
        outer = _ring(self.exterior, True)
        holes = tuple(sorted(_ring(h, False) for h in self.holes))
        polygon = Polygon(outer, holes)
        if not polygon.is_valid or not math.isfinite(polygon.area) or polygon.area <= 0:
            raise ValueError("invalid footprint: " + explain_validity(polygon))
        # Disallow even point contacts: later extruded shells must be manifold.
        rings = [polygon.exterior, *polygon.interiors]
        for i, ring in enumerate(rings):
            if any(ring.intersects(other) for other in rings[:i]):
                raise ValueError("touching footprint rings")
        object.__setattr__(self, "exterior", outer)
        object.__setattr__(self, "holes", holes)

    def polygon(self):
        return Polygon(self.exterior, self.holes)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("missing or unknown fields")
        return cls(**data)

    def canonical(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self):
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()
