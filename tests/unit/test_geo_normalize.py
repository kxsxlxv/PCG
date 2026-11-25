"""Synthetic fixture tests; no provider data or network."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

from shapely import from_wkb
from pcg.source_snapshot import SourceSnapshot, WorldFrame
from pcg.geo_normalize import normalize, report, feature_id

FIXTURES=Path(__file__).resolve().parents[1]/"fixtures"
BBOX=(39.91,43.42,39.94,43.44)


def fixture(kind):
    path=FIXTURES/("l5_"+kind+".geojson")
    template=SourceSnapshot.from_json((FIXTURES/"l5_synthetic.snapshot.json").read_text())
    return replace(template,source_id="fixture."+kind,uri="urn:pcg:fixture:"+kind,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()),path


def frame():
    return WorldFrame("EPSG:32637","synthetic_local_height",(574000,4809000,0))


class NormalizeTests(unittest.TestCase):
    def run_kind(self,kind):
        snapshot,path=fixture(kind)
        return normalize(snapshot,path,frame(),kind,BBOX)

    def test_osm_accept_reject_and_units(self):
        a,r=self.run_kind("osm")
        self.assertEqual(len(a),4)
        self.assertEqual(len(r),7)
        self.assertEqual(report(a,r)["reasons"],{"duplicate_source_id":2,"expected_finite_2d_coordinates":1,
                          "invalid_geometry":1,"missing_stable_source_id":1,
                          "not_building":1,"outside_aoi_owner":1})
        records={x.source_feature_id:x for x in a}
        self.assertEqual(records["way/103"].geometry_kind,"part")
        self.assertEqual(len(from_wkb(bytes.fromhex(records["relation/102"].local_geometry_wkb)).interiors),1)
        obs={x.attribute:x for x in records["way/101"].observations}
        self.assertEqual(obs["height"].value,7.5)
        self.assertEqual(obs["height"].reference,"top_agl")
        self.assertEqual(obs["floors"].value,2)
        obs={x.attribute:x for x in records["way/104"].observations}
        self.assertEqual(obs["height"].status,"invalid")
        self.assertEqual(obs["floors"].status,"invalid")
        self.assertIsNone(obs["height"].value)
        self.assertTrue(all(x.area_m2>0 for x in a))

    def test_overture_multipart_and_extent(self):
        a,r=self.run_kind("overture_building")
        self.assertEqual((len(a),len(r)),(2,1))
        self.assertEqual(r[0].reason,"underground")
        geom=from_wkb(bytes.fromhex(a[1].local_geometry_wkb))
        self.assertEqual(geom.geom_type,"MultiPolygon")
        self.assertEqual(len(geom.geoms),2)
        a,r=self.run_kind("overture_part")
        self.assertEqual((len(a),len(r)),(1,0))
        self.assertEqual(a[0].geometry_kind,"part")
        obs={x.attribute:x for x in a[0].observations}
        self.assertEqual((obs["height"].value,obs["height"].reference),(6.0,"vertical_extent"))
        self.assertEqual((obs["min_height"].value,obs["min_height"].reference),(3.0,"base_agl"))
        self.assertEqual(json.loads(a[0].raw_json)["properties"]["building_id"],"ov-b")

    def test_order_and_source_identity(self):
        snapshot,path=fixture("osm")
        a,r=normalize(snapshot,path,frame(),"osm",BBOX)
        data=json.loads(path.read_text())
        data["features"].reverse()
        raw=json.dumps(data).encode()
        with TemporaryDirectory() as tmp:
            new=Path(tmp)/"reordered.geojson"
            new.write_bytes(raw)
            snap=replace(snapshot,sha256=hashlib.sha256(raw).hexdigest())
            b,s=normalize(snap,new,frame(),"osm",BBOX)
        self.assertNotEqual(snapshot.digest(),snap.digest())
        self.assertEqual([replace(x,snapshot_digest=snap.digest()) for x in a],list(b))
        self.assertEqual([replace(x,snapshot_digest=snap.digest()) for x in r],list(s))
        self.assertIsNone(feature_id({"id":123},"osm"))
        self.assertEqual(feature_id({"properties":{"@type":"relation","@id":5}},"osm"),"relation/5")

    def test_ownership_preserves_full_polygon(self):
        snapshot,path=fixture("osm")
        a,r=normalize(snapshot,path,frame(),"osm",(39.919,43.42,39.92008,43.44))
        self.assertEqual(len(a),1)
        baseline=self.run_kind("osm")[0]
        original=next(x for x in baseline if x.source_feature_id=="way/101")
        self.assertEqual(a[0].local_geometry_wkb,original.local_geometry_wkb)

