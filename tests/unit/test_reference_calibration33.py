import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
from shapely.geometry import box,mapping
from pcg.reference_grammar import MeasuredProfile,build
from pcg.footprint import Footprint
from reference_calibration33 import contour_features,facade_features
class Calibration33Tests(unittest.TestCase):
    def test_contour_evidence(self):
        p=box(.05,.15,.95,.8).union(box(.2,.8,.36,.95))
        row=dict(id='synthetic',pack='low',span_source_units=[20,10,20],orthogonal_confidence=.99,closed_slices=40,vertical_area_fraction=.8,
          contours=[dict(z0=0,z1=.94,polygon=mapping(p),samples=44),dict(z0=.94,z1=1.,polygon=mapping(box(.3,.3,.5,.5)),samples=4)])
        x=contour_features(row);self.assertIsNotNone(x);self.assertTrue(x['relief']);self.assertIsNotNone(x['roof'])
        row['closed_slices']=0;self.assertIsNone(contour_features(row))
    def test_calibrated_plan(self):
        f=Footprint('calibrated',[(0,0),(54,0),(54,15),(0,15)])
        p=MeasuredProfile(1.1,.48,.47,.29,.055,.68,.24,.08,.08)
        a,b=build(f,'walkup_sections',5,14.5,volume=True,measured=p)
        c,d=build(f,'walkup_sections',5,14.5,volume=True,measured=p)
        old,_=build(f,'walkup_sections',5,14.5,volume=True)
        self.assertEqual(a.mass.shell,old.mass.shell)
        self.assertEqual(a.plan.digest(),c.plan.digest());self.assertEqual(b,d)
        self.assertNotEqual(a.plan.profile.spacing,old.plan.profile.spacing)
        self.assertEqual(a.plan.footprint_digest,f.digest())
        self.assertLessEqual(b.planned_triangles+b.reserved_triangles+len(a.mass.shell.triangles()),1800)
    def test_invalid_prior(self):
        with self.assertRaises(ValueError):MeasuredProfile(1.,.4,.4,.2,relief_over_short_side=2.)
