import unittest
import numpy as np
import trimesh
from pcg.contracts import Rectangle
from pcg.geometry import build_shell

class ShellTests(unittest.TestCase):
    def test_shells(self):
        for roof,rise in [("flat",0),("gable",2)]:
            for w,d,h in [(12,8,6),(3,17,2),(1,1,1)]:
                with self.subTest(roof=roof,w=w,d=d):
                    b=Rectangle("fixture",w,d,h,roof,rise)
                    s=build_shell(b)
                    m=trimesh.Trimesh(vertices=s.vertices,faces=[t for _,t in s.triangles()],process=False)
                    self.assertTrue(m.is_watertight)
                    self.assertTrue(m.is_winding_consistent)
                    self.assertTrue(np.all(m.area_faces>0))
                    self.assertAlmostEqual(m.volume,w*d*(h+rise/2))
                    np.testing.assert_allclose(m.bounds,[[0,0,0],[w,d,h+rise]])
                    self.assertEqual(len(s.vertices)-len(s.edges)+len(s.surfaces),2)
                    self.assertEqual(s,build_shell(b))
                    self.assertEqual(sum(e.role=="ridge" for e in s.edges),int(roof=="gable"))
                    self.assertEqual(len({e.id for e in s.edges}),len(s.edges))
