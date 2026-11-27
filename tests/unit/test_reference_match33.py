import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
import numpy as np
from reference_match33 import mesh_tiers,fit_tiers,atlas_centers
class MatchTests(unittest.TestCase):
    def test_box_reconstruction(self):
        span=np.array([8.,4.,6.])
        t,u,r=mesh_tiers([dict(z0=0.,z1=1.,rect=[0,0,1,1])],span)
        self.assertEqual(len(t),12)
        q=fit_tiers(t,span)
        self.assertEqual(len(q),1)
        np.testing.assert_allclose(q[0]['rect'],[0,0,1,1],atol=.02)
    def test_uv_inverse(self):
        t=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,0.,3.]]])
        uv=np.array([[[0.,0.],[1.,0.],[0.,1.]]]);n=np.array([[0.,-1.,0.]])
        a=atlas_centers(t,uv,n,[[10,10,30,30]],(100,100))
        self.assertEqual(len(a),1)
        np.testing.assert_allclose(a[0]['center'],[.4,0,.6])
        self.assertAlmostEqual(a[0]['width_estimate'],.4)
        self.assertAlmostEqual(a[0]['height_estimate'],.6)

    def test_l_contour(self):
        from reference_contours33 import mesh_contours,section_polygon
        from shapely.geometry import Polygon,mapping
        p=Polygon([(0,0),(1,0),(1,.5),(.5,.5),(.5,1),(0,1)])
        span=np.array([8.,4.,6.])
        t,u,r=mesh_contours([dict(z0=0.,z1=1.,polygon=mapping(p))],span)
        q=section_polygon(t,3.123,span)
        self.assertIsNotNone(q)
        self.assertGreater(q.intersection(p).area/q.union(p).area,.99)
    def test_disconnected_contours(self):
        from reference_contours33 import mesh_contours,section_polygon
        from shapely.geometry import box,mapping,MultiPolygon
        p=MultiPolygon([box(0,0,.4,1),box(.6,0,1,1)])
        span=np.array([8.,4.,6.])
        t,u,r=mesh_contours([dict(z0=0.,z1=1.,polygon=mapping(p))],span)
        q=section_polygon(t,3.123,span)
        self.assertIsNotNone(q)
        self.assertEqual(q.geom_type,'MultiPolygon')
        self.assertGreater(q.intersection(p).area/q.union(p).area,.99)
