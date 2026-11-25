"""Audit an explicitly supplied local source snapshot; no acquisition or fusion."""
import argparse
import json
from collections import Counter
from pathlib import Path

from pcg.source_snapshot import SourceSnapshot, read_verified_geojson


def audit(manifest, payload, output):
    snapshot = SourceSnapshot.from_json(Path(manifest).read_text(encoding="utf-8"))
    data = read_verified_geojson(snapshot, payload)
    counts = Counter()
    for feature in data["features"]:
        g = feature.get("geometry")
        kind = g.get("type") if isinstance(g, dict) else None
        counts[kind if kind in ("Polygon", "MultiPolygon", "Point", "MultiPoint",
                                "LineString", "MultiLineString", "GeometryCollection")
               else "missing_or_unknown"] += 1
    report = {
        "schema_version": 1,
        "compiler_version": "0.1.0",
        "source_id": snapshot.source_id,
        "snapshot_digest": snapshot.digest(),
        "payload_sha256": snapshot.sha256,
        "authorization_declaration": snapshot.authorization,
        "release": snapshot.release,
        "epoch": snapshot.epoch,
        "horizontal_crs": snapshot.horizontal_crs,
        "axis_order": snapshot.axis_order,
        "vertical_reference": snapshot.vertical_reference,
        "feature_count": len(data["features"]),
        "declared_geometry_types": dict(sorted(counts.items())),
        "limitations": ["checksum and envelope validated; geometry and coordinate values not audited",
                        "authorization is caller declaration, not legal review",
                        "no conflation, height resolution, source download or terrain processing"]
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.manifest, args.payload, args.output)


if __name__ == "__main__":
    main()
