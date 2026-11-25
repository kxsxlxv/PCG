import unittest
import numpy as np
from pcg.materials import MATERIALS, resolve_material, texture_maps, metric_uv


class MaterialTests(unittest.TestCase):
    def test_maps(self):
        for recipe in MATERIALS.values():
            with self.subTest(recipe=recipe.id):
                maps=texture_maps(recipe,128)
                again=texture_maps(recipe,128)
                self.assertEqual(set(maps),{"basecolor","normal","orm","height"})
                for k,a in maps.items():
                    self.assertEqual(a.shape,(128,128) if k=="height" else (128,128,3))
                    np.testing.assert_array_equal(a,again[k])
                    self.assertTrue(np.isfinite(a).all() and a.min()>=0 and a.max()<=1)
                np.testing.assert_allclose(np.linalg.norm(maps["normal"]*2-1,axis=-1),1,atol=1e-12)
                np.testing.assert_allclose(maps["orm"][...,2],recipe.metalness)
        self.assertEqual(MATERIALS["metal_roof_red"].metalness,0)
        self.assertEqual(MATERIALS["bare_handle"].metalness,1)
        with self.assertRaises(ValueError):texture_maps(MATERIALS["red_brick"],127)

    def test_metric_uv(self):
        for v in ([(0,0,0),(4,0,0),(4,0,3)],
                  [(0,0,0),(4,0,0),(4,3,2)],
                  [(0,0,1),(4,0,1),(4,3,1)],
                  [(0,0,0),(3,4,0),(3,4,2)]):
            v=np.array(v,dtype=float)
            uv,n=metric_uv(v,2.)
            for i,j in ((0,1),(1,2),(2,0)):
                self.assertAlmostEqual(np.linalg.norm(uv[i]-uv[j])*2,np.linalg.norm(v[i]-v[j]))
            np.testing.assert_allclose(metric_uv(v[[1,2,0]],2.)[0],uv[[1,2,0]],atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(n),1.)
        a,n=metric_uv([(0,0,0),(4,0,0),(4,0,3)],2.)
        b,_=metric_uv([(0,0,0),(4,0,3),(0,0,3)],2.)
        np.testing.assert_allclose(a[[0,2]],b[[0,1]])
        np.testing.assert_allclose(a,[(0,0),(2,0),(2,1.5)])
        with self.assertRaises(ValueError):metric_uv([(0,0,0),(1,0,0),(2,0,0)],2.)
        with self.assertRaises(ValueError):metric_uv([(0,0,0),(1,0,0),(1,1,0)],0.)

    def test_resolver(self):
        self.assertEqual(resolve_material("house","wall").id,"plaster_cream")
        self.assertEqual(resolve_material("apartments","balcony_door").id,"glass_proxy")
        self.assertEqual(resolve_material("office","roof").id,"bitumen_roof")
        with self.assertRaises(ValueError):resolve_material("house","unknown")
