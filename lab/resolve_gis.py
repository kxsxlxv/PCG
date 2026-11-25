"""Synthetic explicit adjudication over the actual L5.2 normalized artifacts."""
from pathlib import Path
import json
from pcg.geo_match import build_graph, SourceKey
from pcg.geo_normalize import digest
from pcg.geo_resolve import Approval, adjudicate
from match_gis import load


def main():
    root=Path("artifacts/l5-normalize")
    out=Path("artifacts/l5-resolve");out.mkdir(parents=True,exist_ok=True)
    a=load(root/"osm.json");b=load(root/"overture_building.json")
    graph=build_graph(a,b)
    # These are explicit original fixture labels, not approval of actual buildings.
    group=next(g for g in graph["groups"] if g["status"]=="isolated_high_candidate")
    assert group["left"]==({"source_id":"fixture.osm","feature_id":"way/101"},)
    approval=Approval(group["group_id"],"synthetic-internal-001",
                      SourceKey("fixture.osm","way/101"),"known identical synthetic footprint")
    result=adjudicate(a,b,[approval],expected_graph_digest=digest(graph))
    assert result==adjudicate(reversed(a),reversed(b),[approval],expected_graph_digest=digest(graph))
    assert len(result["resolved"])==1 and len(result["unapproved_groups"])==2
    resolved=result["resolved"][0]["resolution"]
    assert resolved["status"]=="candidate_ready",resolved
    assert resolved["values"]=={"base_agl_m":0.,"top_agl_m":7.5,"roof_rise_m":1.5,
                                "floors":2,"roof":"hip","profile":"house","eave_agl_m":6.}
    (out/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"approved":len(result["resolved"]),"unapproved":len(result["unapproved_groups"]),
                      "status":resolved["status"],"values":resolved["values"]}))


if __name__=="__main__":
    main()
