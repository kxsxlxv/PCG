"""Independent geodetic/axis/precision contract. No UNIGINE SDK required."""
import json, math
from pathlib import Path
import numpy as np
from pyproj import Transformer, __version__ as proj_version


def run():
    # Projected CRS for flat pilot; ellipsoidal heights only in ECEF test.
    lon0, lat0 = 39.925, 43.43
    forward = Transformer.from_crs(4326, 32637, always_xy=True)
    inverse = Transformer.from_crs(32637, 4326, always_xy=True)
    ecef = Transformer.from_crs(4979, 4978, always_xy=True)
    origin = np.array(forward.transform(lon0, lat0))
    lon, lat = np.meshgrid(np.linspace(39.87,40.03,41), np.linspace(43.38,43.51,41))
    e,n = forward.transform(lon,lat)
    p = np.stack([e.ravel(),n.ravel()], axis=1)
    local = p-origin
    # Random submeter vertex coordinates near projected sample points.
    rng=np.random.default_rng(20261125)
    p += rng.uniform(-.49,.49,p.shape)
    local=p-origin
    tile_origin=np.floor(local/512)*512+origin
    tile_local=p-tile_origin
    def err(q): return float(np.max(np.linalg.norm(q-p,axis=1)))
    absolute_float=err(p.astype('float32').astype('float64'))
    relative_float=err(local.astype('float32').astype('float64')+origin)
    tile_float=err(tile_local.astype('float32').astype('float64')+tile_origin)
    lon2,lat2=inverse.transform(e,n)
    roundtrip=float(np.max(np.hypot(lon2-lon,lat2-lat)))
    # At h=0 ellipsoid surface, a tangent-plane ENU frame curves downward away from origin.
    # Not terrain elevation; do not apply this correction to a flat projected landscape.
    xyz=np.stack(ecef.transform(lon,lat,np.zeros_like(lon)),axis=-1)
    ref=np.array(ecef.transform(lon0,lat0,0))
    phi,lam=np.radians([lat0,lon0]); up=np.array([math.cos(phi)*math.cos(lam),math.cos(phi)*math.sin(lam),math.sin(phi)])
    enu_up=(xyz-ref)@up
    # glTF RH Y-up to engine RH Z-up; apply exactly once, with asset mount orientation explicit.
    R=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    a,b=np.array([1.,2.,3.]),np.array([2.,-1.,4.])
    assert np.linalg.det(R)==1 and np.allclose(np.cross(R@a,R@b),R@np.cross(a,b))
    assert np.array_equal(R.T@R,np.eye(3)) and np.array_equal(R@[0,1,0],[0,0,1])
    assert roundtrip<1e-8 and tile_float<.0001
    return dict(samples=len(p),crs='EPSG:32637',origin_lon_lat=[lon0,lat0],origin_e_n_m=origin.tolist(),bbox_lon_lat=[39.87,43.38,40.03,43.51],roundtrip_max_deg=roundtrip,absolute_float_max_error_m=absolute_float,origin_relative_float_max_error_m=relative_float,tile_local_float_max_error_m=tile_float,enu_up_min_m=float(enu_up.min()),enu_up_max_m=float(enu_up.max()),rotation_gltf_to_z_up=R.tolist(),rotation_determinant=float(np.linalg.det(R)),numpy=np.__version__,pyproj=proj_version,limitations=['synthetic points, no DEM or SDK','float32 storage error only, not renderer precision','ENU ellipsoid sag is not a terrain correction','no vertical datum conversion grids evaluated'])


if __name__=='__main__':
    import sys
    r=run();Path(sys.argv[1] if len(sys.argv)>1 else 'coordinate_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
