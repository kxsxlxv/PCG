"""Cross-source candidate lab over the actual L5.2 normalized fixtures."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import from_wkb

from pcg.geo_normalize import SourceFeature, Observation
from pcg.geo_match import build_graph, key


def load(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema_version"]!=1:
        raise ValueError("unsupported normalized schema")
    return tuple(SourceFeature(**{**f,"observations":tuple(Observation(**o) for o in f["observations"])})
                 for f in data["accepted"])


def main():
    root=Path("artifacts/l5-normalize")
    out=Path("artifacts/l5-match");out.mkdir(parents=True,exist_ok=True)
    a=load(root/"osm.json")
    b=load(root/"overture_building.json")
    result=build_graph(a,b)
    assert result==build_graph(reversed(a),reversed(b))
    assert result["counts"]["left_buildings"]==3
    assert result["counts"]["right_buildings"]==2
    assert result["counts"]["excluded_parts"]==1
    assert result["counts"]["groups"]=={"isolated_high_candidate":1,"review_1to1":1,"unmatched_left":1}
    (out/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lookup={(f.source_id,f.source_feature_id):from_wkb(bytes.fromhex(f.local_geometry_wkb)) for f in a+b}
    fig,ax=plt.subplots(figsize=(11,4.5),dpi=150)
    for records,color,label in ((a,"#286a97","OSM"),(b,"#d97724","Overture")):
        first=True
        for f in records:
            if f.geometry_kind!="building":continue
            g=lookup[(f.source_id,f.source_feature_id)]
            for p in (g.geoms if g.geom_type=="MultiPolygon" else (g,)):
                for ring in (p.exterior,*p.interiors):
                    x,y=ring.xy
                    ax.plot(x,y,color=color,linewidth=2 if label=="OSM" else 1.5,
                            linestyle="-" if label=="OSM" else "--",label=label if first else None)
                    first=False
            p=g.representative_point()
            ax.annotate(f.source_feature_id,(p.x,p.y),xytext=(0,10 if label=="OSM" else -14),
                        textcoords="offset points",ha="center",color=color,fontsize=8)
    for edge in result["edges"]:
        p,q=(lookup[(edge[side]["source_id"],edge[side]["feature_id"])].representative_point()
             for side in ("left","right"))
        ax.plot([p.x,q.x],[p.y,q.y],color="#42906c",marker="o",markersize=3,linewidth=1.2)
        ax.annotate("IoU %.2f / %s"%(edge["iou"],edge["rule"]),
                    ((p.x+q.x)/2,(p.y+q.y)/2),xytext=(0,25),textcoords="offset points",
                    ha="center",fontsize=8,color="#275b45")
    ax.set_aspect("equal");ax.margins(x=.12,y=.75)
    ax.set_xlabel("Local east (m)");ax.set_ylabel("Local north (m)")
    ax.set_title("Synthetic GIS candidate graph — no identity merge")
    ax.legend(loc="upper right");ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"candidate_graph.png");plt.close(fig)
    print(json.dumps(result["counts"]))


if __name__=="__main__":
    main()
