import unittest
from dataclasses import replace
from shapely.geometry import box
from pcg.geo_match import build_graph, SourceKey
from pcg.geo_normalize import SourceFeature, Observation, digest
from pcg.geo_resolve import Approval, adjudicate, resolve_fields


def o(attr,value,unit,reference):
    return Observation(attr,value,unit,reference,attr,str(value),"candidate")


def f(src,fid,obs,bounds=(0,0,10,10)):
    g=box(*bounds)
    return SourceFeature(src,src+"-snapshot",fid,"osm" if src=="a" else "overture_building",
                         "building","world",g.normalize().wkb_hex,g.area,"{}",tuple(obs))


def sample():
    a=f("a","1",[o("height",7.5,"m","top_agl"),o("floors",2,"count","above_ground"),
                 o("roof_height",1.5,"m","roof_vertical_extent"),o("roof_shape","hip","none","source_enum"),
                 o("use","house","none","source_enum")])
    b=f("b","2",[o("height",7.5,"m","vertical_extent"),o("floors",2,"count","above_ground"),
                 o("roof_shape","hipped","none","source_enum"),o("use","residential","none","source_enum")])
    return a,b


class ResolveTests(unittest.TestCase):
    def test_agreed_height_and_explicit_geometry(self):
        a,b=sample()
        graph=build_graph([a],[b]);gd=digest(graph)
        ap=Approval(graph["groups"][0]["group_id"],"internal-001",SourceKey("a","1"),"synthetic fixture")
        result=adjudicate([a],[b],[ap],expected_graph_digest=gd)
        rec=result["resolved"][0];res=rec["resolution"]
        self.assertEqual(rec["local_geometry_wkb"],a.local_geometry_wkb)
        self.assertEqual(rec["building_id"],"internal-001")
        self.assertEqual(res["status"],"candidate_ready")
        self.assertEqual(res["values"],{"base_agl_m":0.,"top_agl_m":7.5,"roof_rise_m":1.5,
            "floors":2,"roof":"hip","profile":"house","eave_agl_m":6.})
        self.assertEqual(res["decisions"]["top_agl_m"]["status"],"agreed")
        self.assertEqual(len(res["notes"]),2)
        self.assertEqual(res,resolve_fields([b,a]))

    def test_conflicts_and_missing_are_not_averaged(self):
        a,b=sample()
        b=replace(b,observations=(o("height",11.,"m","vertical_extent"),
            o("floors",3,"count","above_ground"),o("roof_shape","flat","none","source_enum")))
        x=resolve_fields([a,b])
        for name in ("top_agl_m","floors","roof"):
            self.assertEqual(x["decisions"][name]["status"],"conflict")
            self.assertIsNone(x["values"][name])
        self.assertEqual(x["status"],"review_required")
        self.assertIsNone(x["values"]["eave_agl_m"])
        self.assertEqual(resolve_fields([replace(a,observations=())])["status"],"review_required")

    def test_extent_reference_and_invalid_base(self):
        _,b=sample()
        b=replace(b,observations=b.observations+(o("min_height",3.,"m","base_agl"),))
        x=resolve_fields([b])
        self.assertEqual(x["values"]["top_agl_m"],10.5)
        self.assertIn("elevated_building_not_supported",x["blockers"])
        bad=replace(b.observations[-1],value=None,status="invalid")
        x=resolve_fields([replace(b,observations=b.observations[:-1]+(bad,))])
        self.assertIsNone(x["values"]["top_agl_m"])
        self.assertIn("invalid_base",x["blockers"])

    def test_approval_gate_and_stale_digest(self):
        a,b=sample()
        graph=build_graph([a],[b]);gd=digest(graph)
        ap=Approval(graph["groups"][0]["group_id"],"internal-001",SourceKey("a","1"),"fixture")
        self.assertEqual(len(adjudicate([a],[b],[],expected_graph_digest=gd)["unapproved_groups"]),1)
        with self.assertRaises(ValueError):
            adjudicate([a],[b],[ap],expected_graph_digest="stale")
        with self.assertRaises(ValueError):
            adjudicate([a],[b],[replace(ap,geometry_source=SourceKey("x","x"))],expected_graph_digest=gd)
        with self.assertRaises(ValueError):
            adjudicate([a],[b],[ap,ap],expected_graph_digest=gd)
        b2=f("b","3",[],(0,0,5,10))
        g=build_graph([a],[b,b2])
        ap=replace(ap,group_id=g["groups"][0]["group_id"])
        with self.assertRaisesRegex(ValueError,"conflict adjudication"):
            adjudicate([a],[b,b2],[ap],expected_graph_digest=digest(g))
