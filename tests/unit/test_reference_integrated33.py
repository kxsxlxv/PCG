import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
import numpy as np
from reference_integrated33 import crop_window,assignments
class Integrated33Tests(unittest.TestCase):
    def test_crop(self):
        r=np.full((100,100),180,np.uint8)
        self.assertIsNone(crop_window(r,1.))
        r[25:70,30:80]=0
        q=crop_window(r,1.);self.assertIsNotNone(q)
        self.assertGreater(q[1],.8);self.assertLess(q[1],1.4)
    def test_only_unpaired_apertures_and_alias(self):
        m=np.zeros((2,14,4),np.float32);m[:,0,0]=1;m[:,1]=[0,2,2,0]
        o=np.zeros((5,3,4),np.float32);o[:4,0]=[0,0,1,1]
        o[:,2,0]=[1,2,1,4,4];o[2,2,1]=1
        table=[dict(id=0,role='wall',building='b',facade_id='f'),dict(id=1,role='roof',building='b')]
        c=[dict(index=0,family='x',kind='glazing_patch',aspect=1.),dict(index=1,family='x',kind='blind_wall',aspect=1.)]
        q,r=assignments(m,o,table,c,36)
        np.testing.assert_array_equal(q[:,:3],o)
        np.testing.assert_array_equal(q[:,3,0],[37,0,0,0,0])
        self.assertEqual(r['windows'],1);self.assertEqual(r['wall_slots'],4)
        np.testing.assert_array_equal(q[:,3,1],[38,38,38,38,0])
