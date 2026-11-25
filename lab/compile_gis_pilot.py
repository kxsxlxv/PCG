"""Local-only L5.6 package CLI: prepare candidate graph, then compile approvals.

Run with PYTHONPATH=src. All files are explicitly supplied local paths.
Do not upload real-data artifacts to public CI.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image, ImageDraw
from shapely import from_wkb
from pcg.source_snapshot import SourceSnapshot, WorldFrame, strict_json
from pcg.geo_normalize import normalize, report, digest
from pcg.geo_match import build_graph, SourceKey
from pcg.geo_resolve import Approval, adjudicate
from pcg.terrain import load_dtm
from pcg.pilot_compile import compile_approved
from pcg.materials import MATERIALS, texture_maps
from render_material_uv import export_reload
from render_l2_massing import raster


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def write(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def local(root,value):
    if not isinstance(value,str) or not value.strip() or "://" in value:
        raise ValueError("expected explicit local path")
    return root/value


def load_package(path):
    root=Path(path).resolve().parent
    spec=read(path)
    required={"schema_version","frame","bbox_wsen","sources","dtm","seed"}
    if not isinstance(spec,dict) or set(spec)!=required or type(spec["schema_version"]) is not int or spec["schema_version"]!=1:
        raise ValueError("invalid package schema")
    if type(spec["seed"]) is not int:
        raise ValueError("integer seed required")
    frame=WorldFrame.from_json(local(root,spec["frame"]).read_text(encoding="utf-8"))
    sources=spec["sources"]
    if not isinstance(sources,dict) or not {"osm","overture_building"}<=set(sources)<= {"osm","overture_building","overture_part"}:
        raise ValueError("explicit OSM and Overture layers required")
    def input_pair(entry):
        if not isinstance(entry,dict) or set(entry)!={"manifest","payload"}:
            raise ValueError("input requires manifest and payload")
        snap=SourceSnapshot.from_json(local(root,entry["manifest"]).read_text(encoding="utf-8"))
        return snap,local(root,entry["payload"])
    layers={};snapshots={};reports={}
    for kind,entry in sorted(sources.items()):
        snap,payload=input_pair(entry)
        layers[kind],rejected=normalize(snap,payload,frame,kind,spec["bbox_wsen"])
        reports[kind]=report(layers[kind],rejected)
        snapshots[kind]=json.loads(snap.canonical())
    dtm,path=input_pair(spec["dtm"])
    grid=load_dtm(dtm,path,frame)
    snapshots["dtm"]=json.loads(dtm.canonical())
    if len({s["source_id"] for s in snapshots.values()})!=len(snapshots):
        raise ValueError("duplicate source IDs across package")
    graph=build_graph(layers["osm"],layers["overture_building"])
    # Graph approvals bind the source records; package digest additionally binds terrain,
    # frame, AOI, seed and all source snapshots (not filesystem path strings).
    identity={"schema_version":1,"frame":json.loads(frame.canonical()),"bbox_wsen":spec["bbox_wsen"],
              "seed":spec["seed"],"snapshots":snapshots,"graph_digest":digest(graph)}
    return spec,layers,grid,graph,reports,identity


def prepare(package,out):
    spec,layers,grid,graph,reports,identity=load_package(package)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    result={"schema_version":1,"compiler_version":"0.1.0","package_digest":digest(identity),
            "graph_digest":digest(graph),"identity":identity,"graph":graph,
            "layers":reports,"unresolved_overture_parts":[asdict(f) for f in layers.get("overture_part",())]}
    write(out/"prepare.json",result)
    return result


def parse_approvals(path,identity,graph):
    a=read(path)
    if not isinstance(a,dict) or set(a)!={"schema_version","package_digest","graph_digest","approvals"} or type(a["schema_version"]) is not int or a["schema_version"]!=1:
        raise ValueError("invalid approval schema")
    if a["package_digest"]!=digest(identity) or a["graph_digest"]!=digest(graph):
        raise ValueError("stale package or graph approval")
    if not isinstance(a["approvals"],list):
        raise ValueError("approvals must be a list")
    approvals=[]
    for item in a["approvals"]:
        if not isinstance(item,dict) or set(item)!={"group_id","building_id","geometry_source","reason"}:
            raise ValueError("invalid approval fields")
        approvals.append(Approval(**{**item,"geometry_source":SourceKey(**item["geometry_source"])}))
    return approvals


def export_buildings(package,approval_path,out,*,preview=False):
    spec,layers,grid,graph,reports,identity=load_package(package)
    approvals=parse_approvals(approval_path,identity,graph)
    decisions=adjudicate(layers["osm"],layers["overture_building"],approvals,expected_graph_digest=digest(graph))
    compiled=compile_approved(decisions,grid,seed=spec["seed"])
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/"adjudication.json",decisions)
    write(out/"source_reports.json",reports)
    maps={};records=[]
    for record,day,night in compiled:
        if record["status"]!="compiled":
            records.append(record);continue
        name=hashlib.sha256(record["building_id"].encode()).hexdigest()
        directory=out/name;directory.mkdir()
        for material in sorted(day):
            if material not in maps:
                maps[material]={k:Image.fromarray(np.round(v*255).astype(np.uint8))
                                for k,v in texture_maps(MATERIALS[material]).items()
                                if k in ("basecolor","normal","orm")}
        exported={};depths={}
        lo,hi=np.asarray(record["render_bounds_building_local_z_up"])
        center=(lo+hi)/2
        size=512
        scale=(size-72)/max(np.linalg.norm(hi-lo),1.)
        for mode,groups,is_night in (("day",day,False),("night",night,True)):
            path=directory/(mode+".glb")
            meshes=export_reload(groups,maps,path,night=is_night)
            vertices=np.vstack([m.vertices for m in meshes])
            np.testing.assert_allclose([vertices.min(0),vertices.max(0)],[lo,hi],atol=2e-6,rtol=0)
            exported[mode]={"path":str(path.relative_to(out)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes":path.stat().st_size,"triangles":sum(len(m.faces) for m in meshes)}
            if exported[mode]["triangles"]!=record["triangles"]:
                raise ValueError("GLB triangle count mismatch")
            if preview:
                for label,view,up in (("aerial",(1,-1,1.2),(0,0,1)),("top",(0,0,1),(0,1,0))):
                    depth=np.full((size,size),-np.inf)
                    canvas=np.full((size,size,3),[13,19,29] if is_night else [245,243,238],dtype=np.uint8)
                    for m in meshes:
                        im,z=raster(m,view,up,textured=True,night=is_night,center=center,pixels_per_m=scale,size=size)
                        mask=z>depth
                        canvas[mask]=np.asarray(im)[mask];depth[mask]=z[mask]
                    visible=np.isfinite(depth)
                    assert visible.sum()>100 and not visible[[0,-1],:].any() and not visible[:,[0,-1]].any()
                    if is_night:
                        np.testing.assert_array_equal(visible,np.isfinite(depths[label]))
                        np.testing.assert_allclose(depth[visible],depths[label][visible],atol=1e-5)
                    else:depths[label]=depth.copy()
                    im=Image.fromarray(canvas)
                    ImageDraw.Draw(im).text((10,10),record["building_id"]+" / "+mode+" / "+label,
                                           fill=(230,230,235) if is_night else (0,0,0))
                    im.save(directory/(mode+"_"+label+".png"))
        record={**record,"exports":exported}
        write(directory/"building.json",record)
        records.append(record)
    result={"schema_version":1,"compiler_version":"0.1.0","package_digest":digest(identity),
            "identity":identity,"records":records,"unapproved_groups":decisions["unapproved_groups"],
            "excluded_parts":decisions["excluded_parts"],
            "unresolved_overture_parts":[asdict(f) for f in layers.get("overture_part",())],
            "counts":{"approved":len(records),"compiled":sum(r["status"]=="compiled" for r in records),
                      "review_required":sum(r["status"]!="compiled" for r in records),
                      "unapproved_groups":len(decisions["unapproved_groups"])},
            "limitations":["candidate diagnostic compiler, not real-pilot acceptance",
                           "no foundation mesh, terrain modification, native SDK or HLOD",
                           "individual building GLBs with explicit placement sidecars"]}
    write(out/"report.json",result)
    return result


def synthetic_lab():
    """Original test inputs; never describe these as actual Adler structures."""
    root=Path("artifacts/l5-pilot");root.mkdir(parents=True,exist_ok=False)
    fixtures=Path("tests/fixtures")
    frame=WorldFrame.from_json(Path("artifacts/l5-normalize/frame.json").read_text())
    (root/"frame.json").write_text(frame.canonical())
    template=SourceSnapshot.from_json((fixtures/"l5_synthetic.snapshot.json").read_text())
    sources={}
    for kind in ("osm","overture_building","overture_part"):
        data=read(fixtures/("l5_"+kind+".geojson"))
        if kind=="osm":
            # Independent courtyard office, plus an approved but invalid-attribute singleton
            # already in the earlier fixture. No real-world identity implied.
            def ring(x,y,w,h):
                return [[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]]
            data["features"].append({"type":"Feature","id":"w200","properties":{"building":"office",
                   "height":"9 m","building:levels":"3","roof:shape":"flat"},
                   "geometry":{"type":"Polygon","coordinates":[ring(39.923,43.4302,.00024,.00018),
                                 ring(39.92307,43.43025,.00010,.00008)]}})
        payload=root/(kind+".geojson");write(payload,data)
        snap=replace(template,source_id="fixture."+kind,uri="urn:pcg:fixture:pilot:"+kind,
                     sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
        manifest=root/(kind+".snapshot.json");(manifest).write_text(snap.canonical())
        sources[kind]={"manifest":manifest.name,"payload":payload.name}
    e0,n0=frame.project_lonlat(39.9198,43.4298)
    e1,n1=frame.project_lonlat(39.9235,43.4306)
    width=int(np.ceil(e1-e0))+10;height=int(np.ceil(n1-n0))+10
    x=np.arange(width)+.5;y=height-np.arange(height)-.5
    z=frame.origin[2]+10+.004*x[None,:]+.003*y[:,None]
    path=root/"terrain.tif"
    with rasterio.open(path,"w",driver="GTiff",width=width,height=height,count=1,dtype="float64",
                       crs=frame.horizontal_crs,transform=from_origin(e0,n0+height,1,1),nodata=-9999) as ds:
        ds.write(z,1)
    snap=replace(template,source_id="fixture.dtm",uri="urn:pcg:fixture:pilot:dtm",format="geotiff",
                 horizontal_crs=frame.horizontal_crs,axis_order="east_north",
                 vertical_reference=frame.vertical_datum,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/"dtm.snapshot.json").write_text(snap.canonical())
    package=root/"package.json"
    write(package,{"schema_version":1,"frame":"frame.json","bbox_wsen":[39.91,43.42,39.94,43.44],
                   "sources":sources,"dtm":{"manifest":"dtm.snapshot.json","payload":path.name},"seed":17})
    prepared=prepare(package,root/"prepared")
    approvals=[]
    ids={"way/101":"synthetic-house","way/104":"synthetic-invalid","way/200":"synthetic-office"}
    for group in prepared["graph"]["groups"]:
        if len(group["left"])==1 and group["left"][0]["feature_id"] in ids:
            key=group["left"][0]
            approvals.append({"group_id":group["group_id"],"building_id":ids[key["feature_id"]],
                              "geometry_source":key,"reason":"explicit original synthetic fixture identity"})
    assert len(approvals)==3
    approval=root/"approvals.json"
    write(approval,{"schema_version":1,"package_digest":prepared["package_digest"],
                    "graph_digest":prepared["graph_digest"],"approvals":approvals})
    result=export_buildings(package,approval,root/"compiled",preview=True)
    assert result["counts"]=={"approved":3,"compiled":2,"review_required":1,"unapproved_groups":1},result["counts"]
    assert len(result["unresolved_overture_parts"])==1
    good=[r for r in result["records"] if r["status"]=="compiled"]
    assert [r["building_id"] for r in good]==["synthetic-house","synthetic-office"]
    assert len(good[1]["footprint"]["holes"])==1
    sheet=Image.new("RGB",(1024,1024))
    for row,r in enumerate(good):
        for column,mode in enumerate(("day","night")):
            path=(root/"compiled"/r["exports"][mode]["path"]).with_name(mode+"_aerial.png")
            with Image.open(path) as im:sheet.paste(im,(512*column,512*row))
    sheet.save(root/"all_contact.png")
    print(json.dumps(result["counts"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","compile"))
    p.add_argument("--package",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--approvals")
    p.add_argument("--preview",action="store_true")
    a=p.parse_args()
    if a.mode=="prepare":
        result=prepare(a.package,a.out);print(result["graph_digest"])
    else:
        if not a.approvals:p.error("--approvals required")
        result=export_buildings(a.package,a.approvals,a.out,preview=a.preview)
        print(json.dumps(result["counts"]))


if __name__=="__main__":
    main()
