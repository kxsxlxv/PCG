import unittest
import numpy as np
import trimesh
from pcg.footprint import Footprint
from pcg.architecture import PROFILES, plan_facades
from pcg.assets import plan_assets, asset_triangles, parts


class AssetTests(unittest.TestCase):
    def test_module_topology_bounds_and_reuse(self):
        f=Footprint("block",[(0,0),(12,0),(12,8),(0,8)])
        for profile in PROFILES.values():
            p=plan_facades(f,9,3,profile,party_edges=[(0,1)])
            a=plan_assets(f,p)
            self.assertEqual(a,plan_assets(f,p))
            self.assertEqual(a.architecture_digest,p.digest())
            self.assertLess(len(a.recipes),len(a.placements))
            self.assertEqual(len({x.id for x in a.placements}),len(a.placements))
            recipes={r.id:r for r in a.recipes}
            facades={s.id:s for s in p.facades}
            for r in a.recipes:
                bounds=[]
                counts=[]
                for lod in (0,1):
                    triangles=asset_triangles(r,lod)
                    counts.append(len(triangles))
                    vertices=np.array([v for _,t in triangles for v in t])
                    self.assertTrue(np.isfinite(vertices).all())
                    bounds.append([vertices.min(axis=0),vertices.max(axis=0)])
                    self.assertEqual(len(triangles),12*len(parts(r,lod)))
                    for i in range(0,len(triangles),12):
                        m=trimesh.Trimesh(vertices=vertices[i*3:(i+12)*3],faces=np.arange(36).reshape(-1,3),process=True)
                        self.assertTrue(m.is_watertight)
                        self.assertTrue(m.is_winding_consistent)
                        self.assertGreater(m.volume,0)
                np.testing.assert_allclose(bounds[0],bounds[1],atol=1e-9)
                self.assertGreaterEqual(counts[0],counts[1])
            for x in a.placements:
                s=facades[x.facade_id]
                self.assertFalse(s.party)
                o=next(o for o in s.openings if o.id==x.opening_id)
                r=recipes[x.asset_id]
                np.testing.assert_allclose(x.point((0,0,0)),x.origin)
                self.assertAlmostEqual(np.linalg.det(np.array([(*x.tangent,0),(-x.outward[0],-x.outward[1],0),(0,0,1)])),1)
                if r.family!="balcony":
                    np.testing.assert_allclose(x.origin,s.point(o.u+o.width/2,o.z))
                    self.assertAlmostEqual(r.width,o.width)
                    self.assertAlmostEqual(r.height,o.height)
            self.assertEqual(any(r.family=="balcony" for r in a.recipes),profile.id=="apartments")

    def test_courtyard_and_footprint_identity(self):
        f=Footprint("yard",[(0,0),(12,0),(12,10),(0,10)], [[(3,3),(9,3),(9,7),(3,7)]])
        p=plan_facades(f,9,3,PROFILES["apartments"])
        a=plan_assets(f,p)
        recipes={r.id:r for r in a.recipes}
        facades={s.id:s for s in p.facades}
        self.assertTrue(any(facades[x.facade_id].courtyard for x in a.placements))
        self.assertFalse(any(recipes[x.asset_id].family=="balcony" and facades[x.facade_id].courtyard for x in a.placements))
        with self.assertRaises(ValueError):
            plan_assets(Footprint("other",f.exterior,f.holes),p)
