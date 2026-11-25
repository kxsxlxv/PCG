"""Integration tests of building-local compiler, no external datasets."""
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box, Polygon
from pcg.source_snapshot import WorldFrame, SourceSnapshot
from pcg.terrain import TerrainGrid
from pcg.pilot_compile import compile_building, compile_approved


class PilotCompileTests(unittest.TestCase):
    def setUp(self):
        self.frame=WorldFrame("EPSG:32637","synthetic_local_height",(574000,4809000,100))
        snap=SourceSnapshot.from_json((Path(__file__).resolve().parents[1]/"fixtures/l5_synthetic.snapshot.json").read_text())
        snap=replace(snap,format="geotiff",horizontal_crs=self.frame.horizontal_crs,
                     axis_order="east_north",vertical_reference=self.frame.vertical_datum)
        heights=np.full((80,80),110.)
        self.grid=TerrainGrid(snap,self.frame,from_origin(574000,4809080,1,1),
                              heights,np.ones_like(heights,dtype=bool))

    def record(self,bid="house",g=None,roof="hip"):
        if g is None:g=box(5,5,17,15)
        rise=1.5 if roof!="flat" else 0.
        return {"building_id":bid,"group_id":"group-"+bid,"members":[],"geometry_source":{"source_id":"fixture","feature_id":bid},
                "world_frame_digest":self.frame.digest(),"local_geometry_wkb":g.wkb_hex,
                "area_m2":g.area,"resolution":{"status":"candidate_ready","values":{
                    "base_agl_m":0.,"eave_agl_m":6.,"top_agl_m":6.+rise,
                    "roof_rise_m":rise,"roof":roof,"floors":2,"profile":"house"}}}

    def test_compiler_preserves_geometry_and_datum(self):
        source=self.record()
        rec,day,night=compile_building(source,self.grid,seed=17)
        self.assertEqual(rec["status"],"compiled",rec)
        self.assertEqual(rec["anchor_local_z_up"],[5.,5.,10.])
        self.assertEqual(rec["anchor_projected_enh"],[574005.,4809005.,110.])
        self.assertEqual(rec["render_bounds_building_local_z_up"][1][2],7.5)
        self.assertEqual(sum(map(len,day.values())),rec["triangles"])
        self.assertEqual(sum(map(len,night.values())),rec["triangles"])
        self.assertEqual((rec,day,night),compile_building(source,self.grid,seed=17))
        self.assertGreater(rec["semantic_volume_m3"],0)

    def test_holes_and_explicit_unsupported(self):
        outer=box(20,20,40,40)
        hole=box(26,26,34,34)
        g=Polygon(outer.exterior.coords,[hole.exterior.coords])
        rec,_,_=compile_building(self.record("court",g,"flat"),self.grid,seed=17)
        self.assertEqual(rec["status"],"compiled",rec)
        self.assertEqual(len(rec["footprint"]["holes"]),1)
        self.assertAlmostEqual(rec["semantic_volume_m3"],g.area*6.)
        for roof,reason in (("hip","roof_fallback_requires_review"),("gable","roof_facade_integration_unsupported")):
            rec,day,night=compile_building(self.record("court",g,roof),self.grid,seed=17)
            self.assertEqual(rec["status"],"review_required")
            self.assertTrue(rec["reason"].startswith(reason),rec)
            self.assertFalse(day or night)

    def test_overlap_and_deterministic_batch(self):
        a=self.record("a")
        b=self.record("b",box(10,5,22,15))
        c=self.record("c",box(40,40,52,50))
        x=compile_approved({"resolved":[a,b,c]},self.grid,seed=17)
        self.assertEqual([r["status"] for r,_,_ in x],["review_required","review_required","compiled"])
        self.assertEqual(x[0][0]["conflicts"],["b"])
        self.assertEqual(x[1][0]["conflicts"],["a"])
        self.assertEqual(x,compile_approved({"resolved":[c,b,a]},self.grid,seed=17))
