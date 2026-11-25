import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pcg.source_snapshot import SourceSnapshot, WorldFrame, read_verified_geojson, strict_json

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = SourceSnapshot.from_json((FIXTURES/"l5_synthetic.snapshot.json").read_text())
        self.payload = FIXTURES/"l5_synthetic.geojson"

    def test_pinned_local_fixture(self):
        data = read_verified_geojson(self.snapshot, self.payload)
        self.assertEqual(len(data["features"]), 1)
        self.assertEqual(data["features"][0]["id"], "synthetic-a")
        self.assertEqual(self.snapshot, SourceSnapshot.from_json(self.snapshot.canonical()))
        self.assertEqual(self.snapshot.digest(), SourceSnapshot.from_json(self.snapshot.canonical()).digest())
        with TemporaryDirectory() as tmp:
            copied=Path(tmp)/"copy.geojson"
            copied.write_bytes(self.payload.read_bytes())
            self.assertEqual(read_verified_geojson(self.snapshot, copied),data)
            copied.write_bytes(copied.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                read_verified_geojson(self.snapshot,copied)

    def test_manifest_validation(self):
        invalid=({"authorization":"unknown"},{"sha256":"0"*63},{"epoch":"2026-02-30"},
                 {"retrieved_at":"2026-11-25T10:00:00"},{"uri":"https://example.org/file?token=abc"},
                 {"horizontal_crs":"EPSG:3857"},{"axis_order":"lat_lon"},
                 {"vertical_reference":"unknown"},{"license":""},{"schema_version":True})
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.snapshot,**fields)
        d=json.loads(self.snapshot.canonical())
        del d["release"]
        with self.assertRaises(ValueError):
            SourceSnapshot.from_json(json.dumps(d))
        with self.assertRaises(ValueError):
            SourceSnapshot.from_json(self.snapshot.canonical()[:-1]+',"extra":1}')
        for bad in ('{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                strict_json(bad)

    def test_checked_payload_format(self):
        invalid=(b'{"type":"FeatureCollection","features":[],"crs":{}}',
                 b'{"type":"FeatureCollection","features":[{}]}',
                 b'{"type":"Polygon","coordinates":[]}')
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"source.json"
            for raw in invalid:
                path.write_bytes(raw)
                snapshot=replace(self.snapshot,sha256=hashlib.sha256(raw).hexdigest())
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    read_verified_geojson(snapshot,path)


class WorldFrameTests(unittest.TestCase):
    def test_metric_projection_and_origin(self):
        frame=WorldFrame("EPSG:32637","synthetic_local_height",(500000,0,100))
        e,n=frame.project_lonlat(39,0)
        self.assertAlmostEqual(e,500000,places=6)
        self.assertAlmostEqual(n,0,places=6)
        self.assertEqual(frame.local(500001.25,2.5,104,vertical_datum="synthetic_local_height"),
                         (1.25,2.5,4.0))
        self.assertEqual(frame,WorldFrame.from_json(frame.canonical()))
        self.assertEqual(frame.digest(),WorldFrame.from_json(frame.canonical()).digest())
        with self.assertRaisesRegex(ValueError,"vertical datum mismatch"):
            frame.local(e,n,0,vertical_datum="EGM2008")
        with self.assertRaises(ValueError):
            frame.project_lonlat(181,43)

    def test_invalid_frame(self):
        for kwargs in ({"horizontal_crs":"EPSG:4326"},{"horizontal_crs":"EPSG:2263"},
                       {"vertical_datum":"unknown"},{"origin":(0,0,float("inf"))},
                       {"mode":"curved"},{"units":"ft"},{"schema_version":2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(WorldFrame("EPSG:32637","EGM2008",(0,0,0)),**kwargs)
