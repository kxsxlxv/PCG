"""L5.6 offline compiler integration; geometry remains engine-independent.

Building-local render coordinates avoid large world-coordinate float32 loss.
World placement and source lineage remain in the sidecar, not in GLB vertices.
"""
from dataclasses import asdict
import math
import numpy as np
import trimesh
from shapely import from_wkb
from shapely.geometry import box
from shapely.errors import GEOSException
from .footprint import Footprint
from .architecture import PROFILES, plan_facades
from .massing import build_massing
from .facade_mesh import facade_triangles
from .assets import plan_assets, asset_triangles
from .materials import resolve_material
from .occupancy import plan_night, mean_emission
from .terrain import foundation_plan
from .geo_normalize import digest


def compile_building(record, grid, *, seed, spacing_m=2., max_fill_m=3.):
    """Return report + day/night material triangle groups. No implicit repair/fallback."""
    base=foundation_plan(record,grid,spacing_m=spacing_m,max_fill_m=max_fill_m)
    result={"building_id":record["building_id"],"source_group":record["group_id"],
            "members":record["members"],"geometry_source":record["geometry_source"],
            "foundation":base,"resolution":record["resolution"]}
    def reject(reason):
        return {**result,"status":"review_required","reason":reason},{},{}
    if base["status"]!="candidate_ready":
        return reject(base["reason"])
    v=record["resolution"]["values"]
    if v["roof"] not in ("flat","hip"):
        return reject("roof_facade_integration_unsupported")
    g=from_wkb(bytes.fromhex(record["local_geometry_wkb"]))
    ox,oy=g.bounds[:2]
    try:
        f=Footprint(record["building_id"],
                    [(x-ox,y-oy) for x,y in g.exterior.coords],
                    [[(x-ox,y-oy) for x,y in ring.coords] for ring in g.interiors])
        if not math.isclose(f.polygon().area,g.area,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError("footprint area changed")
        h=v["eave_agl_m"]
        mass=build_massing(f,h,v["roof"],v["roof_rise_m"])
        if mass.actual_roof!=v["roof"]:
            return reject("roof_fallback_requires_review:"+str(mass.fallback_reason))
        shell=mass.shell
        # Welded semantic shell checks, independently of split render triangles.
        sm=trimesh.Trimesh(vertices=shell.vertices,faces=[t for _,t in shell.triangles()],process=False)
        if not sm.is_watertight or not sm.is_winding_consistent or sm.volume<=0:
            raise ValueError("invalid semantic shell")
        plan=plan_facades(f,h,v["floors"],PROFILES[v["profile"]])
        assets=plan_assets(f,plan)
        roles={s.id:s.role for s in shell.surfaces}
        tris=[(roles[k],tuple(shell.vertices[i] for i in t),None)
              for k,t in shell.triangles() if roles[k]!="wall"]
        for facade in plan.facades:
            tris.extend((t.role,t.vertices,t.opening_id) for t in facade_triangles(facade,h,0))
        cache={r.id:asset_triangles(r,0) for r in assets.recipes}
        tris.extend((role,tuple(p.point(v) for v in t),None)
                    for p in assets.placements for role,t in cache[p.asset_id])
        xyz=np.asarray([t for _,t,_ in tris],dtype=np.float64)
        if not np.isfinite(xyz).all() or np.any(np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)<=1e-12):
            raise ValueError("nonfinite or degenerate render triangles")
        lo=xyz.reshape(-1,3).min(0);hi=xyz.reshape(-1,3).max(0)
        np.testing.assert_allclose([lo[2],hi[2]],[0,v["top_agl_m"]],atol=1e-8,rtol=0)
        intents=plan_night(plan,seed)
        variants={i.opening_id:i.variant for i in intents}
        day={};night={}
        for role,t,oid in tris:
            material=resolve_material(v["profile"],role).id
            if role=="roof" and mass.actual_roof=="hip":material="ceramic_roof"
            day.setdefault(material,[]).append(t)
            variant=variants[oid] if role in ("window","balcony_door") else "off"
            night.setdefault(material if variant=="off" else material+"__"+variant,[]).append(t)
    except (ValueError,GEOSException,AssertionError) as exc:
        return reject("geometry_rejected:"+str(exc))
    z=base["base_local_z"]
    offset=[ox,oy,z]
    absolute=[a+b for a,b in zip(grid.frame.origin,offset)]
    result.update({"status":"compiled","anchor_local_z_up":offset,"anchor_projected_enh":absolute,
                   "coordinate_contract":"glTF Y-up vertices -> (x,-z,y) + anchor_local_z_up",
                   "footprint":asdict(f),"footprint_digest":f.digest(),"semantic_shell":asdict(shell),
                   "architecture_plan":asdict(plan),"plan_digest":plan.digest(),
                   "assets":asdict(assets),"roof":mass.actual_roof,
                   "render_bounds_building_local_z_up":[lo.tolist(),hi.tolist()],
                   "render_bounds_world_local_z_up":[(lo+offset).tolist(),(hi+offset).tolist()],
                   "semantic_volume_m3":float(sm.volume),"triangles":len(tris),
                   "windows":len(intents),"lit":sum(i.variant!="off" for i in intents),
                   "mean_linear_glazing":mean_emission(intents),
                   "limitations":["no foundation geometry; horizontal floor datum only",
                                  "no automatic party walls or overlapping building parts"]})
    return result,day,night


def compile_approved(adjudication,grid,*,seed,spacing_m=2.,max_fill_m=3.):
    """Reject all intersecting render envelopes rather than silently choosing one."""
    if type(seed) is not int:
        raise ValueError("integer seed required")
    records=sorted(adjudication["resolved"],key=lambda r:r["building_id"])
    if len({r["building_id"] for r in records})!=len(records):
        raise ValueError("duplicate internal building IDs")
    candidates=[compile_building(r,grid,seed=seed,spacing_m=spacing_m,max_fill_m=max_fill_m)
                for r in records]
    from shapely.strtree import STRtree
    good=[i for i,(r,_,_) in enumerate(candidates) if r["status"]=="compiled"]
    envelopes=[]
    for i in good:
        lo,hi=candidates[i][0]["render_bounds_world_local_z_up"]
        envelopes.append(box(lo[0],lo[1],hi[0],hi[1]))
    conflicts={}
    tree=STRtree(envelopes)
    for j,g in enumerate(envelopes):
        for k in map(int,tree.query(g,predicate="intersects")):
            if j!=k:
                conflicts.setdefault(good[j],set()).add(candidates[good[k]][0]["building_id"])
    for i,others in conflicts.items():
        rec,_,_=candidates[i]
        candidates[i]=({**rec,"status":"review_required","reason":"overlapping_render_envelopes",
                         "conflicts":sorted(others)},{},{})
    return tuple(candidates)
