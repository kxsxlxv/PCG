import json
from pathlib import Path
import unittest
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pcg.footprint import Footprint
from pcg.massing import build_massing, repair_duplicates

CASES = json.loads((Path(__file__).resolve().parents[1] / "fixtures/l2_massing.json").read_text())


class MassingTests(unittest.TestCase):
    def test_fixtures(self):
        for c in CASES:
            with self.subTest(case=c["id"]):
                f=Footprint(c["id"], c["outer"], c.get("holes",[]))
                h=6.
                result=build_massing(f,h,c["roof"],c.get("rise",0))
                s=result.shell
                m=trimesh.Trimesh(vertices=s.vertices,faces=[t for _,t in s.triangles()],process=False)
                self.assertTrue(m.is_watertight)
                self.assertTrue(m.is_winding_consistent)
                self.assertTrue(np.isfinite(m.vertices).all())
                self.assertTrue((m.area_faces>0).all())
                self.assertEqual(m.euler_number,2-2*len(f.holes))
                self.assertEqual(bool(result.fallback_reason),c.get("fallback",False))
                self.assertEqual(result.actual_roof,"flat" if c.get("fallback") else c["roof"])
                self.assertAlmostEqual(f.polygon().area,c["area"])
                roles={x.id:x.role for x in s.surfaces}
                for role in ("base","roof"):
                    ps=[Polygon([s.vertices[i][:2] for i in t]) for k,t in s.triangles() if roles[k]==role]
                    self.assertAlmostEqual(sum(p.area for p in ps),c["area"],places=7)
                    self.assertLess(unary_union(ps).symmetric_difference(f.polygon()).area,1e-7)
                if result.actual_roof=="flat":
                    self.assertAlmostEqual(m.volume,c["area"]*h)
                    self.assertFalse(any(e.role in ("ridge","hip") for e in s.edges))
                else:
                    self.assertGreater(m.volume,c["area"]*h)
                    self.assertAlmostEqual(m.bounds[1,2],h+c["rise"])
                if c["id"]=="hip_rectangle":
                    self.assertAlmostEqual(m.volume,96*h+2*(96+16)/3)
                if c["id"]=="hip_square":
                    self.assertAlmostEqual(m.volume,64*h+2*64/3)
                if c["id"]=="hip_triangle":
                    self.assertAlmostEqual(m.volume,54*h+2*54/3)
                if c["id"]=="gable_rotated":
                    self.assertAlmostEqual(m.volume,50*(h+1))
                other=Footprint(c["id"],c["outer"][::-1],[p[::-1] for p in c.get("holes",[])[::-1]])
                self.assertEqual(result,build_massing(other,h,c["roof"],c.get("rise",0)))

    def test_repair_exact_duplicates(self):
        f,report=repair_duplicates("repair",[(0,0),(8,0),(8,0),(8,6),(0,6),(0,0),(0,0)])
        self.assertEqual(f.polygon().area,48)
        self.assertEqual(report["removed_consecutive_duplicates"],2)
        self.assertEqual(report["area_delta_m2"],0)
        with self.assertRaises(ValueError):
            repair_duplicates("bowtie",[(0,0),(8,8),(0,8),(8,0)])

    def test_reject_bad_parameters(self):
        f=Footprint("test",[(0,0),(8,0),(0,8)])
        for h,roof,rise in [(0,"flat",0),(6,"flat",1),(6,"hip",0),(6,"gable",-1),(6,"unknown",0),(float("nan"),"flat",0)]:
            with self.subTest(roof=roof,h=h),self.assertRaises(ValueError):
                build_massing(f,h,roof,rise)
