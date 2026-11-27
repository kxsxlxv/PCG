import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
import numpy as np
from pcg.footprint import Footprint
from reference_shared33 import patch_regions,atlas_pack,build
class Shared33Tests(unittest.TestCase):
    def test_patch_regions(self):
        r=np.full((128,256),180,np.uint8)
        for y in (16,48,80):
            for x in (55,110,165):r[y:y+12,x:x+20]=0
        self.assertGreater(len(patch_regions(r,[8,40,72,104],1.)),3)
    def test_arbitrary_footprint(self):
        m=[(dict(kind='glazing_patch',ratio=1.,family='test'),[np.full((8,8,3),v,np.uint8) for v in (80,128,180)]) for _ in range(3)]
        _,cat=atlas_pack(m)
        f=Footprint('court',[(0,0),(30,0),(30,24),(0,24)], [[(6,6),(6,18),(24,18),(24,6)]])
        family=dict(glazing=[0],blind=1,roof=2,ratio=1.)
        t,u,r=build(f,6,18,family,cat,((0,1),))
        self.assertLessEqual(len(t),1800);self.assertEqual(r['roof_area'],f.polygon().area)
        self.assertTrue(np.all((u>=0)&(u<=1)))
