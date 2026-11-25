import unittest
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from pcg.footprint import Footprint
from pcg.architecture import PROFILES, Profile, plan_facades
from pcg.facade_mesh import facade_triangles


class ArchitectureTests(unittest.TestCase):
    def test_profiles_and_apertures(self):
        f=Footprint("block",[(0,0),(12,0),(12,8),(0,8)])
        for profile in PROFILES.values():
            with self.subTest(profile=profile.id):
                plan=plan_facades(f,9,3,profile,party_edges=[(0,1)])
                self.assertEqual(plan.floor_levels,(0,3,6,9))
                self.assertEqual(plan,plan_facades(Footprint("block",f.exterior[::-1]),9,3,profile,party_edges=[(0,1)]))
                self.assertEqual(len({o.id for s in plan.facades for o in s.openings}),sum(len(s.openings) for s in plan.facades))
                self.assertTrue(plan.facades[1].party)
                self.assertFalse(plan.facades[1].openings)
                self.assertEqual(sum(o.role=="door" for s in plan.facades for o in s.openings),1)
                for s in plan.facades:
                    for o in s.openings:
                        self.assertGreater(o.u,0)
                        self.assertLess(o.u+o.width,s.length)
                        self.assertGreater(o.z,plan.floor_levels[o.floor])
                        self.assertLess(o.z+o.height,plan.floor_levels[o.floor+1])
                    full=facade_triangles(s,9,0)
                    simple=facade_triangles(s,9,1)
                    self.assertEqual({t.opening_id for t in full},{t.opening_id for t in simple})
                    self.assertEqual([t for t in full if t.role=="wall"],[t for t in simple if t.role=="wall"])
                    self.assertEqual(len(full)-len(simple),8*len(s.openings))
                    self.assertEqual(full,facade_triangles(s,9,0))
                    uv=np.array(s.end)-s.start
                    uv=uv/s.length
                    def project(p):
                        return (float((np.array(p[:2])-s.start)@uv),p[2])
                    polys=[Polygon([project(p) for p in t.vertices]) for t in full if t.role=="wall"]
                    apertures=[box(o.u,o.z,o.u+o.width,o.z+o.height) for o in s.openings]
                    expected=box(0,0,s.length,9).difference(unary_union(apertures))
                    self.assertLess(unary_union(polys).symmetric_difference(expected).area,1e-8)
                    self.assertAlmostEqual(sum(p.area for p in polys),expected.area)
                    for t in full:
                        a,b,c=np.array(t.vertices)
                        self.assertGreater(np.linalg.norm(np.cross(b-a,c-a)),1e-10)
                        if t.role in ("window","door"):
                            self.assertTrue(all(abs((np.array(p[:2])-s.start)@s.outward+profile.recess)<1e-9 for p in t.vertices))

    def test_courtyard_orientation_and_exclusion(self):
        f=Footprint("yard",[(0,0),(12,0),(12,10),(0,10)], [[(3,3),(9,3),(9,7),(3,7)]])
        p=plan_facades(f,6,2,PROFILES["house"],party_edges=[(0,2)])
        self.assertEqual(sum(s.courtyard for s in p.facades),4)
        for s in p.facades:
            from shapely.geometry import Point
            mid=s.point(s.length/2,0,-.01)
            self.assertFalse(f.polygon().contains(Point(mid[:2])))
            mid=s.point(s.length/2,0,.01)
            self.assertTrue(f.polygon().contains(Point(mid[:2])))
            if s.courtyard:self.assertFalse(any(o.role=="door" for o in s.openings))
        self.assertEqual(p.digest(),plan_facades(f,6,2,PROFILES["house"],party_edges=[(0,2)]).digest())

    def test_reject_and_short_facades(self):
        f=Footprint("small",[(0,0),(1,0),(1,1),(0,1)])
        self.assertTrue(all(not s.openings for s in plan_facades(f,2,1,PROFILES["office"]).facades))
        for kwargs in ({"party_edges":[(0,5)]},{"entrance_edge":(0,4)},{"party_edges":[(0,0)],"entrance_edge":(0,0)}):
            with self.assertRaises(ValueError):plan_facades(f,3,1,PROFILES["house"],**kwargs)
        with self.assertRaises(ValueError):plan_facades(f,3,True,PROFILES["house"])
        with self.assertRaises(ValueError):Profile("bad",2,1,1,1,1,1,1)
