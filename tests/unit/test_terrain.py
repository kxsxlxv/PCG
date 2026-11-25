import hashlib
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from pcg.source_snapshot import SourceSnapshot, WorldFrame
from pcg.terrain import load_dtm, foundation_plan

FIXTURES=Path(__file__).resolve().parents[1]/"fixtures"


def write_grid(path,frame,*,nodata_cell=False,steep=False,crs=None):
    # 20 x 20 at 1 m, heights at pixel centers. Local z = 10 + ax + by.
    x=np.arange(20)+.5;y=19.5-np.arange(20)
    z=frame.origin[2]+10+(.8 if steep else .1)*x[None,:]+.05*y[:,None]
    if nodata_cell:z[10,10]=-9999
    with rasterio.open(path,"w",driver="GTiff",width=20,height=20,count=1,
                       dtype="float64",crs=crs or frame.horizontal_crs,
                       transform=from_origin(frame.origin[0],frame.origin[1]+20,1,1),
                       nodata=-9999) as ds:
        ds.write(z,1)
    template=SourceSnapshot.from_json((FIXTURES/"l5_synthetic.snapshot.json").read_text())
    return replace(template,source_id="fixture.dtm",uri="urn:pcg:fixture:dtm",
                   format="geotiff",horizontal_crs=frame.horizontal_crs,axis_order="east_north",
                   vertical_reference=frame.vertical_datum,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class TerrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"dtm.tif"
        self.frame=WorldFrame("EPSG:32637","synthetic_local_height",(574000,4809000,100))

    def record(self,bounds=(3,3,13,13)):
        return {"building_id":"synthetic","world_frame_digest":self.frame.digest(),
                "local_geometry_wkb":box(*bounds).wkb_hex,
                "resolution":{"status":"candidate_ready","values":{"eave_agl_m":6,"top_agl_m":7.5}}}

    def test_bilinear_and_base(self):
        snap=write_grid(self.path,self.frame)
        g=load_dtm(snap,self.path,self.frame)
        self.assertAlmostEqual(g.sample(3.25,8.75),10+.325+.4375,places=8)
        self.assertAlmostEqual(g.sample(.5,.5),10+.05+.025,places=8)
        self.assertIsNone(g.sample(.1,.1))
        p=foundation_plan(self.record(),g)
        self.assertEqual(p["status"],"candidate_ready")
        self.assertAlmostEqual(p["base_local_z"],11.95,places=8)
        self.assertAlmostEqual(p["sampled_relief_m"],1.5,places=8)
        self.assertAlmostEqual(p["eave_local_z"],17.95,places=8)
        self.assertAlmostEqual(p["top_local_z"],19.45,places=8)
        self.assertEqual(p,foundation_plan(self.record(),g))
        self.assertFalse(g.heights.flags.writeable)

    def test_missing_and_steep(self):
        snap=write_grid(self.path,self.frame,nodata_cell=True)
        g=load_dtm(snap,self.path,self.frame)
        self.assertIsNone(g.sample(10.5,9.5))
        self.assertIsNone(g.sample(10,10))
        self.assertEqual(foundation_plan(self.record((8,8,12,12)),g,spacing_m=1)["reason"],"terrain_coverage_missing")
        self.assertEqual(foundation_plan(self.record((0,0,2,2)),g)["reason"],"terrain_coverage_missing")
        snap=write_grid(self.path,self.frame,steep=True)
        g=load_dtm(snap,self.path,self.frame)
        self.assertEqual(foundation_plan(self.record(),g)["reason"],"sampled_relief_exceeds_policy")

    def test_metadata_and_checksum(self):
        snap=write_grid(self.path,self.frame)
        with self.assertRaisesRegex(ValueError,"vertical datum mismatch"):
            load_dtm(snap,self.path,replace(self.frame,vertical_datum="EGM2008"))
        with self.assertRaisesRegex(ValueError,"horizontal CRS mismatch"):
            load_dtm(snap,self.path,replace(self.frame,horizontal_crs="EPSG:32638"))
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            load_dtm(replace(snap,sha256="0"*64),self.path,self.frame)
        snap=write_grid(self.path,self.frame,crs="EPSG:32638")
        with self.assertRaisesRegex(ValueError,"raster CRS mismatch"):
            load_dtm(snap,self.path,self.frame)
