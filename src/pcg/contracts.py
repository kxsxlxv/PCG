"""Strict initial local-metric rectangle contract; no GIS inference."""
from dataclasses import dataclass, asdict
import math
import json
import hashlib

def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name}: expected finite number")
    return float(value)

@dataclass(frozen=True)
class Rectangle:
    building_id: str
    width_m: float
    depth_m: float
    eave_m: float
    roof: str
    roof_rise_m: float
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
        for name in ("width_m", "depth_m", "eave_m", "roof_rise_m"):
            value = number(getattr(self, name), name)
            if value < 0 or (name != "roof_rise_m" and value == 0):
                raise ValueError(f"{name}: invalid dimension")
            object.__setattr__(self, name, value)
        if self.roof not in ("flat", "gable"):
            raise ValueError("unsupported roof")
        if (self.roof == "flat") != (self.roof_rise_m == 0):
            raise ValueError("roof rise inconsistent with roof type")

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("expected object")
        required = set(cls.__dataclass_fields__)
        if set(data) != required:
            raise ValueError("missing or unknown fields")
        return cls(**data)

    def canonical(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self):
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()
