"""Bridge approved synthetic GIS pilot to the neutral zero-opening-mesh ABI.

Do not reinterpret unresolved source records, change geometry, or claim foundation
construction. The pilot report and its accepted architecture are the comparison
authority, not an inferred real-world dataset.
"""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
from compile_zero_tiles import SCHEMA, build_dataset, dump, sha

def convert(report, tile_size):
    f=report["identity"]["frame"]
    if (f["axis_order"],f["units"],f["mode"])!=("east_north_up","m","flat"):
        raise ValueError("unsupported pilot coordinate frame")
    data={"schema":SCHEMA,"world_frame":{"horizontal_crs":f["horizontal_crs"],
        "vertical_datum":f["vertical_datum"],"origin":f["origin"],"units":"m","axis":"ENH"},
        "tile_size_m":tile_size,"buildings":[]}
    accepted={};excluded=[]
    for r in report["records"]:
        bid=r["building_id"]
        if r["status"]!="compiled":
            excluded.append({"building_id":bid,"status":r["status"],"reason":r["reason"],
                             "source_group":r["source_group"]})
            continue
        if bid in accepted:raise ValueError("duplicate approved building")
        v=r["resolution"]["values"];p=r["footprint"];a=r["anchor_local_z_up"]
        assert p["building_id"]==bid and r["architecture_plan"]["building_id"]==bid
        assert p["frame"]=="local_z_up" and p["units"]=="m"
        assert r["foundation"]["status"]=="candidate_ready"
        assert v["base_agl_m"]==0 and abs(a[2]-r["foundation"]["base_local_z"])<1e-8
        assert v["roof"]==r["roof"]
        np.testing.assert_allclose(np.asarray(a)+f["origin"],r["anchor_projected_enh"],atol=1e-8,rtol=0)
        # This bridge deliberately supports only the party-wall-free pilot.
        if any(w["party"] for w in r["architecture_plan"]["facades"]):
            raise ValueError("pilot party-wall adapter not implemented")
        data["buildings"].append({"id":bid,"outer":p["exterior"],"holes":p["holes"],
            "anchor":a,"roof":v["roof"],"rise_m":v["roof_rise_m"],"eave_m":v["eave_agl_m"],
            "floors":v["floors"],"profile":v["profile"],"party_edges":[],
            "source_ref":json.dumps({"package_digest":report["package_digest"],
                 "geometry_source":r["geometry_source"],"source_group":r["source_group"],
                 "members":r["members"]},sort_keys=True,separators=(",",":"))})
        accepted[bid]=r
    assert len(accepted)==report["counts"]["compiled"]
    assert len(accepted)+len(excluded)==len(report["records"])
    return data,accepted,excluded

def triangle_error(source,output):
    # Match unordered triangles, then their unordered vertices; compare the
    # complete semantic shell rather than only bounds or triangle count.
    assert source.shape==output.shape
    used=set();error=0.
    perms=list(itertools.permutations(range(3)))
    for t in source:
        best=(float("inf"),-1)
        for i,s in enumerate(output):
            if i in used:continue
            e=min(float(np.max(np.abs(t-s[list(p)]))) for p in perms)
            if e<best[0]:best=(e,i)
        assert best[0]<5e-5,best
        used.add(best[1]);error=max(error,best[0])
    assert len(used)==len(source)
    return error

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot",type=Path,default=Path("artifacts/l5-pilot/compiled/report.json"))
    ap.add_argument("--library",type=Path,default=Path("artifacts/zero-multitile/library/manifest.json"))
    ap.add_argument("--out",type=Path,default=Path("artifacts/zero-approved-pilot"))
    ap.add_argument("--decoder",type=Path,required=True)
    args=ap.parse_args()
    report=json.loads(args.pilot.read_text())
    data,accepted,excluded=convert(report,128)
    # Explicit check: unresolved status cannot silently enter the export.
    bad=json.loads(json.dumps(report));bad["records"][0]["foundation"]["status"]="review_required"
    try:convert(bad,128)
    except (ValueError,AssertionError):pass
    else:raise AssertionError("unapproved foundation accepted")
    args.out.mkdir(parents=True,exist_ok=True);dump(args.out/"input.json",data)
    result,geometry,semantic=build_dataset(data,args.out/"tiles",args.library,args.decoder.resolve(),preview=False)
    idx=json.loads((args.out/"tiles/index.json").read_text())
    rows=[];seen=set()
    for t in idx["tiles"]:
        m=json.loads((args.out/"tiles"/t["manifest"]).read_text())
        for b in m["owned_buildings"]:
            bid=b["building_id"];assert bid not in seen;seen.add(bid);r=accepted[bid]
            assert b["footprint"]==r["footprint"]
            assert b["roof"]==r["roof"] and b["fallback_reason"] is None
            assert b["plan_digest"]==r["plan_digest"]
            assert b["openings"]==sum(len(w["openings"]) for w in r["architecture_plan"]["facades"])
            shell=r["semantic_shell"];v=np.asarray(shell["vertices"],dtype=np.float64)+r["anchor_local_z_up"]
            tri=np.asarray([v[[s["vertices"][0],s["vertices"][j],s["vertices"][j+1]]]
                        for s in shell["surfaces"] for j in range(1,len(s["vertices"])-1)])
            assert b["triangles"]==len(tri)
            e=triangle_error(tri,geometry[bid].reshape(-1,3,3))
            world=np.asarray(m["world_anchor_enh"])+b["offset_local"]
            np.testing.assert_allclose(world,r["anchor_projected_enh"],rtol=0,atol=1e-8)
            rows.append({"building_id":bid,"original_detailed_triangles":r["triangles"],
              "semantic_shell_triangles":len(tri),"zero_opening_mesh_triangles":b["triangles"],
              "openings":b["openings"],"max_triangle_coordinate_error_m":e,
              "plan_digest":b["plan_digest"],"foundation":r["foundation"],
              "source_group":r["source_group"],"geometry_source":r["geometry_source"],"members":r["members"]})
    assert seen==set(accepted)==set(geometry)
    assert result["opening_triangles_added"]==0
    summary={"source_report_sha256":sha(args.pilot),"source_package_digest":report["package_digest"],
       "fixture_only":True,"approved_buildings":len(rows),"excluded_records":excluded,
       "unapproved_groups":report["unapproved_groups"],"excluded_parts":report["excluded_parts"],
       "unresolved_overture_parts":report["unresolved_overture_parts"],"rows":rows,
       "old_detailed_triangles":sum(r["original_detailed_triangles"] for r in rows),
       "new_shell_triangles":sum(r["zero_opening_mesh_triangles"] for r in rows),
       "export":result,"same_canonical_plan":True,"same_semantic_shell":True,
       "limitations":["synthetic approved pilot only; no real geospatial acceptance",
          "triangle reduction includes removal of physical facade details; not equal visual detail",
          "DTM sampled placement preserved, not foundation geometry",
          "no native SDK, real GPU benchmark or runtime streaming"]}
    dump(args.out/"report.json",summary)
    print("APPROVED_PILOT_REPORT "+json.dumps({k:v for k,v in summary.items() if k not in ("rows","unresolved_overture_parts")}),flush=True)
    print("APPROVED_PILOT_BUILDINGS "+json.dumps([{k:v for k,v in r.items() if k!="foundation"} for r in rows]),flush=True)

if __name__=="__main__":main()
