"""Verified local DTM adapter. No reprojection, vertical conversion or terrain generation."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from pyproj import CRS
from shapely import from_wkb, segmentize
from shapely.geometry import Point
from .source_snapshot import SourceSnapshot, WorldFrame
from .contracts import number

MAX_BYTES=128*1024*1024
MAX_CELLS=16_000_000


@dataclass(frozen=True)
class TerrainGrid:
    snapshot: SourceSnapshot
    frame: WorldFrame
    affine: object
    heights: np.ndarray
    valid: np.ndarray

    def sample(self,x,y):
        """Bilinear sampling at local XY; pixel-center convention, no extrapolation."""
        e=number(x,"x")+self.frame.origin[0]
        n=number(y,"y")+self.frame.origin[1]
        col,row=(~self.affine)*(e,n)
        col-=.5;row-=.5
        h,w=self.heights.shape
        # Allow numerical noise from affine roundtrip at exact pixel centers.
        if not (-1e-8<=col<=w-1+1e-8 and -1e-8<=row<=h-1+1e-8):
            return None
        col=min(max(col,0.),w-1.);row=min(max(row,0.),h-1.)
        c0=int(math.floor(col));r0=int(math.floor(row))
        c1=min(c0+1,w-1);r1=min(r0+1,h-1)
        dx=col-c0;dy=row-r0
        result=0.
        for r,c,weight in ((r0,c0,(1-dx)*(1-dy)),(r0,c1,dx*(1-dy)),
                           (r1,c0,(1-dx)*dy),(r1,c1,dx*dy)):
            if weight<=1e-12:continue
            if not self.valid[r,c]:return None
            result+=weight*float(self.heights[r,c])
        return result-self.frame.origin[2]


def load_dtm(snapshot:SourceSnapshot,path,frame:WorldFrame):
    if not isinstance(snapshot,SourceSnapshot) or snapshot.format!="geotiff":
        raise ValueError("expected GeoTIFF snapshot")
    if snapshot.vertical_reference!=frame.vertical_datum:
        raise ValueError("vertical datum mismatch")
    if CRS.from_user_input(snapshot.horizontal_crs)!=CRS.from_user_input(frame.horizontal_crs):
        raise ValueError("horizontal CRS mismatch; pre-align terrain explicitly")
    with Path(path).open("rb") as f:
        raw=f.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError("DTM byte budget exceeded")
    if hashlib.sha256(raw).hexdigest()!=snapshot.sha256:raise ValueError("DTM checksum mismatch")
    if raw[:4] not in (b"II*\x00",b"MM\x00*",b"II+\x00",b"MM\x00+"):
        raise ValueError("expected TIFF signature")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",PROJ_NETWORK="OFF"):
        with MemoryFile(raw) as mem,mem.open(driver="GTiff") as ds:
            if ds.driver!="GTiff" or ds.count!=1 or ds.width<2 or ds.height<2 or ds.width*ds.height>MAX_CELLS:
                raise ValueError("unsupported DTM raster dimensions or band count")
            if ds.crs is None or CRS.from_user_input(ds.crs.to_wkt())!=CRS.from_user_input(frame.horizontal_crs):
                raise ValueError("raster CRS mismatch")
            if ds.gcps[0] or ds.rpcs is not None:
                raise ValueError("GCP/RPC terrain unsupported")
            t=ds.transform
            if not all(math.isfinite(v) for v in tuple(t)) or t.b!=0 or t.d!=0 or t.a<=0 or t.e>=0:
                raise ValueError("expected north-up affine DTM")
            if ds.scales!=(1.0,) or ds.offsets!=(0.0,):
                raise ValueError("scaled DTM must be explicitly normalized")
            if ds.units[0] not in (None,"m","metre","meter"):
                raise ValueError("expected elevations in metres")
            if np.dtype(ds.dtypes[0]).kind not in "iuf":
                raise ValueError("unsupported elevation dtype")
            a=ds.read(1,masked=True).astype(np.float64)
            values=np.asarray(a.data).copy()
            valid=(~np.ma.getmaskarray(a))&np.isfinite(values)
    values.setflags(write=False);valid.setflags(write=False)
    return TerrainGrid(snapshot,frame,t,values,valid)


def foundation_plan(record,grid:TerrainGrid,*,spacing_m=2.,max_fill_m=3.):
    """Sampling proposal, not a continuous terrain bound or foundation mesh."""
    spacing=number(spacing_m,"spacing_m")
    limit=number(max_fill_m,"max_fill_m")
    if not .25<=spacing<=20 or not 0<=limit<=20:
        raise ValueError("invalid sampling policy")
    if record["world_frame_digest"]!=grid.frame.digest():
        raise ValueError("building world frame mismatch")
    result={"building_id":record["building_id"],"snapshot_digest":grid.snapshot.digest(),
            "frame_digest":grid.frame.digest(),"spacing_m":spacing,"max_fill_m":limit}
    if record["resolution"]["status"]!="candidate_ready":
        return {**result,"status":"review_required","reason":"unresolved_attributes"}
    g=from_wkb(bytes.fromhex(record["local_geometry_wkb"]))
    if g.geom_type!="Polygon" or not g.is_valid or g.is_empty:
        return {**result,"status":"review_required","reason":"unsupported_footprint"}
    lo,bo,hi,to=g.bounds
    nx=math.ceil((hi-lo)/spacing);ny=math.ceil((to-bo)/spacing)
    if nx*ny+g.length/spacing>100000:
        return {**result,"status":"review_required","reason":"sample_budget_exceeded"}
    dense=segmentize(g,spacing)
    points=set((float(x),float(y)) for ring in (dense.exterior,*dense.interiors) for x,y in ring.coords)
    points.add(tuple(g.representative_point().coords[0]))
    for j in range(ny):
        y=bo+(j+.5)*spacing
        for i in range(nx):
            x=lo+(i+.5)*spacing
            if g.covers(Point(x,y)):points.add((x,y))
    samples=[{"x":x,"y":y,"z":grid.sample(x,y)} for x,y in sorted(points)]
    result["samples"]=samples
    result["sample_count"]=len(samples)
    result["missing_count"]=sum(s["z"] is None for s in samples)
    if result["missing_count"]:
        return {**result,"status":"review_required","reason":"terrain_coverage_missing"}
    low=min(s["z"] for s in samples);high=max(s["z"] for s in samples)
    result.update({"sampled_min_z":low,"sampled_max_z":high,"sampled_relief_m":high-low})
    if high-low>limit:
        return {**result,"status":"review_required","reason":"sampled_relief_exceeds_policy"}
    # A horizontal floor datum, not wall tilting or an engineering design.
    result.update({"status":"candidate_ready","base_local_z":high,
                   "eave_local_z":high+record["resolution"]["values"]["eave_agl_m"],
                   "top_local_z":high+record["resolution"]["values"]["top_agl_m"],
                   "limitations":["sampled extrema only","no foundation mesh, excavation or structural assessment"]})
    return result
