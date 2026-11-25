"""Immutable local source provenance and flat metric world frame.

L5.1 supports explicit local RFC 7946 GeoJSON snapshots only. URI is provenance,
never a download target. A caller's authorization declaration is not legal review.
"""
from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from pyproj import CRS, Transformer, network
from .contracts import number


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value

    def invalid_constant(token):
        raise ValueError("nonfinite JSON constant: " + token)

    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=invalid_constant)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(name + ": expected nonempty trimmed text")
    return value


def _version(value):
    if type(value) is not int or value != 1:
        raise ValueError("unsupported schema version")


def _canonical(record):
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(record):
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def _from_json(cls, raw):
    data = strict_json(raw)
    if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
        raise ValueError("missing or unknown fields")
    return cls(**data)


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    uri: str
    release: str
    epoch: str
    license: str
    access_terms: str
    authorization: str
    sha256: str
    format: str
    horizontal_crs: str
    axis_order: str
    vertical_reference: str
    retrieved_at: str
    schema_version: int = 1

    def __post_init__(self):
        _version(self.schema_version)
        for key in ("source_id", "uri", "release", "epoch", "license", "access_terms",
                    "vertical_reference", "retrieved_at"):
            _text(getattr(self, key), key)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", self.source_id):
            raise ValueError("invalid source_id")
        uri = urlsplit(self.uri)
        if not uri.scheme or uri.username is not None or uri.password is not None or uri.query or uri.fragment:
            raise ValueError("URI must be provenance without credentials, query or fragment")
        if self.authorization not in ("synthetic_fixture", "authorized_local"):
            raise ValueError("explicit local authorization declaration required")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("expected lowercase SHA-256")
        if self.format != "geojson" or self.horizontal_crs not in ("OGC:CRS84", "EPSG:4326") or self.axis_order != "lon_lat":
            raise ValueError("L5.1 supports WGS84 lon/lat GeoJSON only")
        # Snapshot epoch is a declared ISO date or explicitly unknown; retrieval
        # date is distinct and must contain a timezone. No invented capture date.
        if self.epoch != "unknown":
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.epoch):
                raise ValueError("expected ISO epoch date or unknown")
            date.fromisoformat(self.epoch)
        if "T" not in self.retrieved_at:
            raise ValueError("retrieved_at must be an ISO timestamp")
        when = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        if when.utcoffset() is None:
            raise ValueError("retrieved_at requires timezone")
        if self.vertical_reference.lower() in ("unknown", "unspecified", "none", "null"):
            raise ValueError("explicit vertical reference required; use not_applicable_2d for 2D source")

    @classmethod
    def from_json(cls, raw):
        return _from_json(cls, raw)

    def canonical(self):
        return _canonical(self)

    def digest(self):
        return _digest(self)


def read_verified_geojson(snapshot: SourceSnapshot, payload_path):
    """Check exact bytes before parsing the same byte buffer; no network access."""
    if not isinstance(snapshot, SourceSnapshot):
        raise ValueError("expected SourceSnapshot")
    raw = Path(payload_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != snapshot.sha256:
        raise ValueError("source checksum mismatch")
    data = strict_json(raw.decode("utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("expected GeoJSON FeatureCollection")
    if "crs" in data:
        raise ValueError("legacy GeoJSON crs member unsupported; normalize explicitly")
    if any(not isinstance(f, dict) or f.get("type") != "Feature" for f in data["features"]):
        raise ValueError("invalid FeatureCollection entry")
    return data


@dataclass(frozen=True)
class WorldFrame:
    horizontal_crs: str
    vertical_datum: str
    origin: tuple
    mode: str = "flat"
    axis_order: str = "east_north_up"
    units: str = "m"
    transform_revision: int = 1
    schema_version: int = 1

    def __post_init__(self):
        _version(self.schema_version)
        _version(self.transform_revision)
        _text(self.horizontal_crs, "horizontal_crs")
        _text(self.vertical_datum, "vertical_datum")
        if self.vertical_datum.lower() in ("unknown", "unspecified", "none", "null", "not_applicable_2d"):
            raise ValueError("world vertical datum must be explicit")
        if self.mode != "flat" or self.axis_order != "east_north_up" or self.units != "m":
            raise ValueError("expected flat metric ENU")
        crs = CRS.from_user_input(self.horizontal_crs)
        axes = crs.axis_info
        if not crs.is_projected or len(axes) != 2 or tuple(a.direction for a in axes) != ("east", "north") or any(a.unit_conversion_factor != 1.0 for a in axes):
            raise ValueError("expected projected east/north metre CRS")
        if not isinstance(self.origin, (tuple, list)) or len(self.origin) != 3:
            raise ValueError("origin must contain E,N,H")
        object.__setattr__(self, "origin", tuple(number(v, "origin") for v in self.origin))

    def project_lonlat(self, lon, lat):
        lon, lat = number(lon, "longitude"), number(lat, "latitude")
        if not (-180 <= lon <= 180 and -90 < lat < 90):
            raise ValueError("invalid longitude/latitude")
        if network.is_network_enabled():
            raise ValueError("PROJ network access must be disabled for offline projection")
        transformer = Transformer.from_crs("OGC:CRS84", self.horizontal_crs,
                                           always_xy=True, allow_ballpark=False)
        east, north = transformer.transform(lon, lat, errcheck=True)
        return number(east, "easting"), number(north, "northing")

    def local(self, east, north, height, *, vertical_datum):
        """Subtract origin from already aligned ENH; never assume datum equality."""
        if vertical_datum != self.vertical_datum:
            raise ValueError("vertical datum mismatch; convert explicitly")
        values = (number(east, "easting"), number(north, "northing"), number(height, "height"))
        return tuple(number(v-o, "local coordinate") for v, o in zip(values, self.origin))

    @classmethod
    def from_json(cls, raw):
        return _from_json(cls, raw)

    def canonical(self):
        return _canonical(self)

    def digest(self):
        return _digest(self)
