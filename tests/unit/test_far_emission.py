import unittest
from dataclasses import replace
import numpy as np
from pcg.architecture import PROFILES, plan_facades
from pcg.footprint import Footprint
from pcg.occupancy import plan_night, WindowIntent, EMISSION
from pcg.far_emission import facade_emission, linear_mips, srgb_encode, srgb_decode


class FarEmissionTests(unittest.TestCase):
    def test_all_facades_preserve_analytic_mean(self):
        for name, profile in PROFILES.items():
            f=Footprint(name,[(0,0),(12,0),(12,8),(0,8)])
            plan=plan_facades(f,9,3,profile,party_edges=((0,1),))
            intents=plan_night(plan)
            by_id={i.opening_id:i for i in intents}
            for s in plan.facades:
                expected=np.zeros(3)
                for o in s.openings:
                    if o.id in by_id:
                        expected+=o.width*o.height*np.asarray(by_id[o.id].emissive_linear)/(s.length*9)
                for size in (8,32,128):
                    mask=facade_emission(s,9,intents,size)
                    rev=replace(s,openings=tuple(reversed(s.openings)))
                    np.testing.assert_array_equal(mask,facade_emission(rev,9,tuple(reversed(intents)),size))
                    chain=linear_mips(mask)
                    self.assertEqual(chain[-1].shape,(1,1,3))
                    for level in chain:
                        np.testing.assert_allclose(level.mean((0,1)),expected,atol=1e-12,rtol=0)
                        quant=srgb_decode(np.round(srgb_encode(level)*255)/255)
                        self.assertLessEqual(np.max(np.abs(quant.mean((0,1))-expected)),.0045)

    def test_subpixel_window_and_rejections(self):
        f=Footprint("x",[(0,0),(12,0),(12,8),(0,8)])
        p=plan_facades(f,6,2,PROFILES["house"])
        s=next(s for s in p.facades if any(o.role=="window" for o in s.openings))
        o=next(o for o in s.openings if o.role=="window")
        o=replace(o,u=.123,z=.234,width=.001,height=.002)
        s=replace(s,openings=(o,))
        i=WindowIntent(o.id,o.width*o.height,"warm")
        for n in (2,16,128):
            a=facade_emission(s,6,(i,),n)
            expected=np.asarray(EMISSION["warm"])*i.area_m2/(s.length*6)
            np.testing.assert_allclose(linear_mips(a)[-1][0,0],expected,atol=1e-14,rtol=0)
            self.assertGreater(a.max(),0)
        with self.assertRaises(ValueError):
            facade_emission(s,6,(),16)
        with self.assertRaises(ValueError):
            facade_emission(s,6,(i,i),16)
        with self.assertRaises(ValueError):
            facade_emission(s,6,(i,),15)
        with self.assertRaises(ValueError):
            facade_emission(s,6,(replace(i,area_m2=1),),16)
        with self.assertRaises(ValueError):
            linear_mips(np.zeros((3,3,3)))
