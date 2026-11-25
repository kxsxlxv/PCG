"""Local GIS normalization CLI and synthetic geometry diagnostic."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from shapely import from_wkb

from pcg.source_snapshot import SourceSnapshot, WorldFrame
from pcg.geo_normalize import normalize, report


def run(manifest,payload,frame,kind,bbox,output):
    snapshot=SourceSnapshot.from_json(Path(manifest).read_text(encoding="utf-8"))
    world=WorldFrame.from_json(Path(frame).read_text(encoding="utf-8"))
    accepted,rejected=normalize(snapshot,payload,world,kind,bbox)
    result=report(accepted,rejected)
    result.update({"snapshot_digest":snapshot.digest(),"world_frame_digest":world.digest(),
                   "source_kind":kind,"bbox_wsen":list(bbox),
                   "authorization_declaration":snapshot.authorization})
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    return result


def synthetic_lab():
    root=Path("tests/fixtures")
    out=Path("artifacts/l5-normalize")
    out.mkdir(parents=True,exist_ok=True)
    frame=WorldFrame("EPSG:32637","synthetic_local_height",(574000,4809000,0))
    frame_path=out/"frame.json"
    frame_path.write_text(frame.canonical(),encoding="utf-8")
    bbox=(39.91,43.42,39.94,43.44)
    template=SourceSnapshot.from_json((root/"l5_synthetic.snapshot.json").read_text())
    reports={}
    for kind in ("osm","overture_building","overture_part"):
        path=root/("l5_"+kind+".geojson")
        snapshot=replace(template,source_id="fixture."+kind,uri="urn:pcg:fixture:"+kind,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest=out/(kind+".snapshot.json")
        manifest.write_text(snapshot.canonical(),encoding="utf-8")
        reports[kind]=run(manifest,path,frame_path,kind,bbox,out/(kind+".json"))
    assert [reports[k]["counts"]["accepted"] for k in reports]==[4,2,1]
    assert [reports[k]["counts"]["rejected"] for k in reports]==[7,1,0]
    # Fixed metric viewport, drawn from normalized WKB, not fabricated layout.
    e,n=frame.project_lonlat(39.92125,43.43015)
    cx,cy=e-frame.origin[0],n-frame.origin[1]
    width,height=900,420
    scale=2.0
    bg=(245,243,238)
    sheet=Image.new("RGB",(width,2*height),bg)
    for row,(title,kinds) in enumerate((("OSM",("osm",)),("Overture",("overture_building","overture_part")))):
        im=Image.new("RGB",(width,height),bg)
        draw=ImageDraw.Draw(im)
        draw.text((12,12),title+" / synthetic source geometry / fixed 2 px per metre",fill=(30,35,40))
        def px(coords):
            pts=[(width/2+(x-cx)*scale,height/2-(y-cy)*scale) for x,y in coords]
            assert all(6<x<width-6 and 30<y<height-6 for x,y in pts)
            return pts
        for kind in kinds:
            for record in sorted(reports[kind]["accepted"],key=lambda r:r["geometry_kind"]=="part"):
                g=from_wkb(bytes.fromhex(record["local_geometry_wkb"]))
                part=record["geometry_kind"]=="part"
                fill=(225,154,75) if part else (91,155,174)
                for p in (g.geoms if g.geom_type=="MultiPolygon" else (g,)):
                    draw.polygon(px(p.exterior.coords),fill=fill,outline=(40,65,74),width=1)
                    for ring in p.interiors:
                        draw.polygon(px(ring.coords),fill=bg,outline=(40,65,74),width=1)
                x,y=px([g.representative_point().coords[0]])[0]
                draw.text((x+4,y+4),record["source_feature_id"],fill=(20,30,35))
        draw.line((20,height-28,120,height-28),fill=(30,35,40),width=2)
        draw.text((20,height-23),"50 m",fill=(30,35,40))
        draw.text((width-235,height-28),"blue: building / orange: part",fill=(30,35,40))
        sheet.paste(im,(0,row*height))
    sheet.save(out/"normalized_geometry.png")
    summary={"schema_version":1,"compiler_version":"0.1.0","synthetic":True,
             "layers":{k:v["counts"] for k,v in reports.items()},
             "rejections":{k:v["reasons"] for k,v in reports.items()}}
    (out/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--payload",required=True)
    p.add_argument("--frame",required=True)
    p.add_argument("--kind",choices=("osm","overture_building","overture_part"),required=True)
    p.add_argument("--bbox",nargs=4,type=float,required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    result=run(a.manifest,a.payload,a.frame,a.kind,a.bbox,a.output)
    print(json.dumps(result["counts"]))


if __name__=="__main__":
    main()
