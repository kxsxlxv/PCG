import unittest
from dataclasses import replace
from shapely.geometry import box
from pcg.geo_normalize import SourceFeature
from pcg.geo_match import build_graph, MatchPolicy


def feature(source, fid, bounds, kind="building"):
    g=box(*bounds)
    return SourceFeature(source,source+"-snapshot",fid,"osm" if source=="a" else "overture_building",
                         kind,"metric-frame",g.normalize().wkb_hex,g.area,"{}",())


class MatchGraphTests(unittest.TestCase):
    def test_components_and_exclusions(self):
        a=[feature("a","exact",(0,0,10,10)),feature("a","split",(30,0,50,10)),
           feature("a","near",(70,0,80,10)),feature("a","touch",(100,0,110,10)),
           feature("a","alone",(140,0,150,10)),feature("a","part",(0,0,5,5),"part")]
        b=[feature("b","exact",(0,0,10,10)),feature("b","split1",(30,0,40,10)),
           feature("b","split2",(40,0,50,10)),feature("b","near",(82,0,92,10)),
           feature("b","touch",(110,0,120,10))]
        result=build_graph(a,b)
        self.assertEqual(result["counts"]["groups"],{"ambiguous_1toN":1,"isolated_high_candidate":1,
            "review_1to1":1,"unmatched_left":2,"unmatched_right":1})
        self.assertEqual(result["counts"]["edges"],4)
        self.assertEqual(result["counts"]["boundary_touches"],1)
        self.assertEqual(result["counts"]["excluded_parts"],1)
        self.assertEqual(result,build_graph(reversed(a),reversed(b)))
        self.assertEqual(sum(len(g["left"]) for g in result["groups"]),5)
        self.assertEqual(sum(len(g["right"]) for g in result["groups"]),5)

    def test_many_to_many_is_not_merge(self):
        a=[feature("a","a1",(0,0,10,10)),feature("a","a2",(10,0,20,10))]
        b=[feature("b","b1",(0,0,20,5)),feature("b","b2",(0,5,20,10))]
        x=build_graph(a,b)
        self.assertEqual(x["counts"]["groups"],{"ambiguous_NtoM":1})
        self.assertEqual(len(x["edges"]),4)
        self.assertEqual(len(x["groups"][0]["left"]),2)
        self.assertEqual(len(x["groups"][0]["right"]),2)

    def test_validation_and_empty(self):
        a=feature("a","a",(0,0,10,10));b=feature("b","b",(0,0,10,10))
        self.assertEqual(build_graph([a],[])["counts"]["groups"],{"unmatched_left":1})
        self.assertEqual(build_graph([],[])["counts"]["edges"],0)
        for left,right in (([a,a],[b]),([a],[replace(b,world_frame_digest="other")]),
                           ([a],[a]),([replace(a,area_m2=1)],[b])):
            with self.assertRaises(ValueError):
                build_graph(left,right)
        with self.assertRaises(ValueError):
            MatchPolicy(gap_m=-1)

    def test_containment_and_weak_overlap(self):
        a=[feature("a","large",(0,0,20,20)),feature("a","weak",(40,0,50,10))]
        b=[feature("b","small",(2,2,6,6)),feature("b","weak",(49,0,59,10))]
        x=build_graph(a,b)
        self.assertEqual(x["edges"][0]["rule"],"containment_review")
        self.assertEqual(x["counts"]["weak_overlaps"],1)
        self.assertEqual(x["counts"]["groups"],{"review_1to1":1,"unmatched_left":1,"unmatched_right":1})
