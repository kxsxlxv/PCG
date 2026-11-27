import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
import numpy as np
from reference_pbr33 import basis,project,fill_nearest,floor_strips,recompose,glb_write
class PBR33Tests(unittest.TestCase):
    def test_projection(self):
        for view in ([1,0,0],[-1,0,0],[0,0,1],[0,0,-1]):
            r,u,v=basis(view);np.testing.assert_allclose(np.cross(r,u),v)
            np.testing.assert_allclose(project(np.array([r,u]),np.zeros(3),[1,1],view),[[1,.5],[.5,0]])
    def test_fill(self):
        a=np.zeros((5,5,3),np.uint8);m=np.zeros((5,5),bool);m[2,2]=True;a[2,2]=[12,34,56]
        np.testing.assert_array_equal(fill_nearest(a,m,[0,0,0]),np.broadcast_to([12,34,56],a.shape))
    def test_repeated_floors(self):
        r=np.full((256,128,3),180,np.uint8);mask=np.ones((256,128),bool)
        for y in range(16,250,32):r[y:y+9,10:118,1]=0
        q=floor_strips(r,mask);self.assertIsNotNone(q);self.assertLessEqual(abs(q['pitch_px']-32),1)
        a,c=recompose([r,r,r],q,12,mask);self.assertGreater(c,0);np.testing.assert_array_equal(a[0],a[1])
    def test_glb_roundtrip(self):
        t=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,0.,3.]]]);u=np.array([[[.1,.1],[.8,.1],[.1,.8]]])
        maps=[np.full((8,8,3),v,np.uint8) for v in (120,128,200)]
        with tempfile.TemporaryDirectory() as td:glb_write(Path(td)/'a.glb',t,u,maps)
