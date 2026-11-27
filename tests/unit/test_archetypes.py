import math
import unittest
from collections import Counter
import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from pcg.footprint import Footprint
from pcg.architecture import PROFILES, plan_facades, BALCONY_DEPTH, BALCONY_WIDTH_EXTRA
from pcg.archetypes import ARCHETYPES, plan_variant, select_archetype, MAX_BALCONY_TRIANGLES


def rect(key,w=30,d=14):
    return Footprint(key,[(0,0),(w,0),(w,d),(0,d)])


class ArchetypeTests(unittest.TestCase):
    def test_all_presets_geometry_and_budget(self):
        self.assertEqual(len(ARCHETYPES),15)
        roofs=set();signatures=set()
        for a in ARCHETYPES.values():
            with self.subTest(archetype=a.id):
                floors,top,w,d={"house":(2,9.5,14,10),"apartments":(5,18.5,30,14),"office":(4,14.8,24,16)}[a.use]
                f=rect(a.id,w,d)
                v=plan_variant(f,a.use,floors,top,archetype=a.id)
                self.assertEqual(v.plan.footprint_digest,f.digest())
                self.assertEqual(v.plan.floor_levels[-1],v.eave_m)
                self.assertEqual(len(v.plan.floor_levels),floors+1)
                self.assertAlmostEqual(max(p[2] for p in v.mass.shell.vertices),top,places=8)
                s=v.mass.shell
                m=trimesh.Trimesh(vertices=s.vertices,faces=[t for _,t in s.triangles()],process=False)
                self.assertTrue(m.is_watertight and m.is_winding_consistent and m.volume>0)
                self.assertTrue(np.all(m.area_faces>1e-10))
                roles={x.id:x.role for x in s.surfaces}
                rp=[Polygon([s.vertices[i][:2] for i in t]) for sid,t in s.triangles() if roles[sid]=="roof"]
                self.assertLess(unary_union(rp).symmetric_difference(f.polygon()).area,1e-8)
                self.assertAlmostEqual(sum(p.area for p in rp),f.polygon().area)
                openings=[o for wall in v.plan.facades for o in wall.openings]
                self.assertEqual(len(openings),len({o.id for o in openings}))
                self.assertEqual(sum(o.role=="door" for o in openings),1)
                self.assertGreater(sum(o.role=="window" for o in openings),0)
                balc=[]
                for wall in v.plan.facades:
                    rs=[]
                    for o in wall.openings:
                        self.assertTrue(0<o.u<o.u+o.width<wall.length)
                        self.assertTrue(v.plan.floor_levels[o.floor]<o.z<o.z+o.height<v.plan.floor_levels[o.floor+1])
                        r=box(o.u,o.z,o.u+o.width,o.z+o.height)
                        self.assertFalse(any(r.intersects(q) for q in rs));rs.append(r)
                        if o.role=="balcony_door":
                            self.assertFalse(wall.courtyard or wall.party or o.floor==0)
                            w=o.width+BALCONY_WIDTH_EXTRA;u=o.u+o.width/2
                            poly=Polygon([wall.point(x,0,depth)[:2] for x,depth in ((u-w/2,0),(u+w/2,0),(u+w/2,-BALCONY_DEPTH),(u-w/2,-BALCONY_DEPTH))])
                            self.assertLess(poly.intersection(f.polygon()).area,1e-8)
                            for q,l in balc:
                                if l==o.floor:self.assertLess(q.intersection(poly).area,1e-8)
                            balc.append((poly,o.floor))
                self.assertLessEqual(len(s.triangles())+len(balc)*MAX_BALCONY_TRIANGLES,v.max_triangles)
                rev=Footprint(f.building_id,f.exterior[::-1])
                self.assertEqual(v,plan_variant(rev,a.use,floors,top,archetype=a.id))
                roofs.add(v.mass.actual_roof)
                signatures.add((v.mass.actual_roof,a.dimensions,a.composition,a.balcony))
        self.assertEqual(roofs,{"flat","gable","hip"})
        self.assertEqual(len(signatures),15)

    def test_regional_prior_and_selection_contract(self):
        ids=[f"stable-{i}" for i in range(500)]
        for region in ("baltic","mediterranean","caucasus"):
            vals=[select_archetype(k,"apartments",5,region) for k in ids]
            self.assertEqual(vals,[select_archetype(k,"apartments",5,region) for k in ids])
            self.assertEqual(vals,[select_archetype(k,"apartments",5,region) for k in reversed(ids)][::-1])
            neutral=[select_archetype(k,"apartments",5) for k in ids]
            self.assertGreater(sum(region in a.regions for a in vals),sum(region in a.regions for a in neutral))
        for args in (("x","office",4,"unknown"),("x","house",False,"generic"),("x","house",20,"generic")):
            with self.assertRaises(ValueError):select_archetype(*args)
        with self.assertRaises(ValueError):select_archetype("x","office",5,override="panel_slab")

    def test_observed_roof_and_fallback(self):
        f=rect("explicit",14,10)
        v=plan_variant(f,"house",2,9.2,archetype="baltic_timber",roof="hip",eave_m=6.7)
        self.assertEqual((v.mass.actual_roof,v.eave_m,v.roof_source,v.eave_source),("hip",6.7,"observed","observed"))
        self.assertAlmostEqual(max(p[2] for p in v.mass.shell.vertices),9.2)
        v=plan_variant(f,"house",2,9.2,archetype="baltic_timber",roof="flat")
        self.assertEqual((v.mass.actual_roof,v.eave_m),("flat",9.2))
        for kw in ({"roof":"flat","eave_m":6.7},{"roof":"gable","eave_m":9.2},{"roof":"mansard"},{"eave_m":10.}):
            with self.assertRaises(ValueError):plan_variant(f,"house",2,9.2,**kw)
        l=Footprint("L",[(0,0),(14,0),(14,5),(6,5),(6,10),(0,10)])
        v=plan_variant(l,"house",2,9.2,archetype="baltic_timber")
        self.assertEqual((v.mass.requested_roof,v.mass.actual_roof,v.eave_m),("gable","flat",9.2))
        self.assertTrue(v.mass.fallback_reason)
        with self.assertRaisesRegex(ValueError,"requires review"):
            plan_variant(l,"house",2,9.2,archetype="baltic_timber",roof="gable",eave_m=6.7)
        v=plan_variant(f,"house",2,5.4,archetype="nordic_cottage")
        self.assertEqual(v.mass.fallback_reason,"insufficient_height_for_inferred_pitch")
        self.assertEqual(v.eave_m,5.4)

    def test_budget_complete_columns_and_party_courtyard(self):
        f=rect("large",80,16)
        v=plan_variant(f,"apartments",10,33,archetype="panel_slab",max_triangles=260)
        cols=Counter((wall.id,o.id.rsplit(":",1)[-1]) for wall in v.plan.facades for o in wall.openings if o.role=="balcony_door")
        self.assertTrue(cols);self.assertEqual(set(cols.values()),{9})
        self.assertGreater(v.budget_omitted_slots,0)
        self.assertLessEqual(len(v.mass.shell.triangles())+sum(cols.values())*20,260)
        v0=plan_variant(f,"apartments",10,33,archetype="panel_slab",max_triangles=100)
        self.assertFalse(any(o.role=="balcony_door" for wall in v0.plan.facades for o in wall.openings))
        yard=Footprint("yard",[(0,0),(30,0),(30,20),(0,20)], [[(8,6),(22,6),(22,14),(8,14)]])
        v=plan_variant(yard,"apartments",5,18.5,archetype="perimeter_block",party_edges=((0,0),))
        self.assertEqual(v.mass.actual_roof,"flat")
        self.assertFalse(v.plan.facades[0].openings)
        self.assertFalse(any(o.role in ("balcony_door","door") for wall in v.plan.facades if wall.courtyard for o in wall.openings))
        with self.assertRaises(ValueError):plan_variant(f,"apartments",10,33,max_triangles=8)

    def test_legacy_and_policy_validation(self):
        f=rect("legacy-stage25",12,8)
        expected={"house":"102d8b1d607c4ded87344899da5592d9bda60ff74502458486a400630d12e24f","apartments":"a1f4fb38d2d7bd59c2e6bfc0b157f7d56dcd8088d53e5973968c3be5322681e3","office":"7039f06f8fe038636efb64dfd2b496df91ea39f7bd346824eb2eaee3bc920fc1"}
        for k,p in PROFILES.items():
            self.assertEqual(plan_facades(f,9,3,p,party_edges=((0,1),)).digest(),expected[k])
        for bad in ("invalid","balcony"):
            with self.assertRaises(ValueError):plan_facades(f,9,3,PROFILES["office"],slot_policy=lambda *_:bad)


if __name__=="__main__":unittest.main()
