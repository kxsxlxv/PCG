import unittest
from dataclasses import replace
from pcg.architecture import PROFILES, plan_facades
from pcg.footprint import Footprint
from pcg.facade_mesh import facade_triangles
from pcg.occupancy import EMISSION, WindowIntent, plan_night, mean_emission


class OccupancyTests(unittest.TestCase):
    def test_stable_identity_and_lod(self):
        for profile, style in PROFILES.items():
            f = Footprint(profile, [(0,0),(12,0),(12,8),(0,8)])
            plan = plan_facades(f, 9, 3, style, party_edges=((0,1),))
            intents = plan_night(plan)
            self.assertEqual(intents, plan_night(plan))
            self.assertEqual(intents, plan_night(replace(plan, facades=tuple(
                replace(s, openings=tuple(reversed(s.openings)))
                for s in reversed(plan.facades)))))
            expected = {o.id for s in plan.facades for o in s.openings
                        if o.role in ("window", "balcony_door")}
            self.assertEqual({i.opening_id for i in intents}, expected)
            for lod in (0,1):
                actual = {t.opening_id for s in plan.facades
                          for t in facade_triangles(s,9,lod)
                          if t.role in ("window", "balcony_door")}
                self.assertEqual(actual, expected)
            self.assertTrue(all(i.variant in EMISSION and i.area_m2 > 0 for i in intents))
            self.assertNotEqual(intents, plan_night(plan, 18))
        with self.assertRaises(ValueError):
            plan_night(plan, True)

    def test_linear_aggregation(self):
        self.assertEqual(mean_emission(()), (0.,0.,0.))
        a = WindowIntent("a", 3., "warm")
        b = WindowIntent("b", 1., "off")
        for actual, expected in zip(mean_emission((a,b)), EMISSION["warm"]):
            self.assertAlmostEqual(actual, .75 * expected)
        self.assertEqual(mean_emission((b,)), (0.,0.,0.))
