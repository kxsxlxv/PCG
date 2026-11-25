"""Synthetic DTM integration on explicitly approved L5.4 source geometry."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely import from_wkb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pcg.source_snapshot import SourceSnapshot, WorldFrame
from pcg.terrain import load_dtm, foundation_plan


def main():
    out=Path("artifacts/l5-dtm");out.mkdir(parents=True,exist_ok=True)
    frame=WorldFrame.from_json(Path("artifacts/l5-normalize/frame.json").read_text())
    report=json.loads(Path("artifacts/l5-resolve/report.json").read_text())
    record=report["resolved"][0]
    g=from_wkb(bytes.fromhex(record["local_geometry_wkb"]))
    xmin,ymin,xmax,ymax=g.bounds
    x0=np.floor(xmin)-5;y1=np.ceil(ymax)+5
    width=int(np.ceil(xmax)+5-x0);height=int(y1-np.floor(ymin)+5)
    x=x0+np.arange(width)+.5;y=y1-np.arange(height)-.5
    def analytic(x,y):
        return 12+.08*(x-x0)+.04*(y-(y1-height))
    z=frame.origin[2]+analytic(x[None,:],y[:,None])
    path=out/"synthetic_dtm.tif"
    with rasterio.open(path,"w",driver="GTiff",width=width,height=height,count=1,
                       dtype="float64",crs=frame.horizontal_crs,
                       transform=from_origin(frame.origin[0]+x0,frame.origin[1]+y1,1,1),
                       nodata=-9999) as ds:
        ds.write(z,1)
    template=SourceSnapshot.from_json(Path("tests/fixtures/l5_synthetic.snapshot.json").read_text())
    snap=replace(template,source_id="fixture.dtm",uri="urn:pcg:fixture:dtm-v1",
                 format="geotiff",horizontal_crs=frame.horizontal_crs,axis_order="east_north",
                 vertical_reference=frame.vertical_datum,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/"snapshot.json").write_text(snap.canonical())
    grid=load_dtm(snap,path,frame)
    plan=foundation_plan(record,grid,spacing_m=1.)
    assert plan["status"]=="candidate_ready",plan
    error=max(abs(s["z"]-analytic(s["x"],s["y"])) for s in plan["samples"])
    assert error<1e-8
    assert plan["base_local_z"]==plan["sampled_max_z"]
    np.testing.assert_allclose([plan["eave_local_z"]-plan["base_local_z"],
                                plan["top_local_z"]-plan["base_local_z"]],[6.,7.5],atol=1e-10)
    result={"schema_version":1,"compiler_version":"0.1.0","synthetic":True,
            "rasterio":rasterio.__version__,"gdal":rasterio.__gdal_version__,
            "max_analytic_error_m":error,"plan":plan}
    (out/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    fig,ax=plt.subplots(figsize=(8,6),dpi=150)
    image=ax.imshow(grid.heights-frame.origin[2],extent=(x0,x0+width,y1-height,y1),
                    origin="upper",cmap="terrain")
    for ring in (g.exterior,*g.interiors):
        xx,yy=ring.xy;ax.plot(xx,yy,color="#222222",linewidth=1.6)
    ax.scatter([s["x"] for s in plan["samples"]],[s["y"] for s in plan["samples"]],
               s=9,color="white",edgecolors="#222222",linewidths=.35,label="Terrain sample")
    ax.set_aspect("equal");ax.legend(loc="upper right")
    ax.set_title("Synthetic terrain and footprint sampling\nHorizontal base %.3f m | sampled relief %.3f m"%(plan["base_local_z"],plan["sampled_relief_m"]))
    ax.set_xlabel("Local east (m)");ax.set_ylabel("Local north (m)")
    fig.colorbar(image,ax=ax,label="Local height (m)")
    fig.tight_layout();fig.savefig(out/"terrain_samples.png");plt.close(fig)
    print(json.dumps({"samples":plan["sample_count"],"relief_m":plan["sampled_relief_m"],
                      "base_local_z":plan["base_local_z"],"max_analytic_error_m":error}))


if __name__=="__main__":
    main()
