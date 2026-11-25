"""Metric material recipes and planar UVs for neutral render meshes.

Original procedural texture kernel from research 06; shared PBR export via
trimesh, not a custom renderer. GLTF normal convention +Y, ORM R/G/B.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np
from ._material_patterns import RECIPES, material as pattern


@dataclass(frozen=True)
class MaterialRecipe:
    id: str
    repeat_m: float
    height_amplitude_m: float
    roughness: float
    metalness: float
    basecolor_srgb: tuple
    pattern_id: str | None
    revision: int = 1

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,
            separators=(",",":"),allow_nan=False).encode()).hexdigest()


MATERIALS = {
    key:MaterialRecipe(key,repeat,amplitude,rough,metal,tuple(c/255 for c in color),key)
    for key,(_,color,repeat,amplitude,rough,metal) in RECIPES.items()
}
MATERIALS.update({
    "glass_proxy":MaterialRecipe("glass_proxy",1.,0.,.18,0.,(.16,.29,.38),None),
    "painted_trim":MaterialRecipe("painted_trim",1.,0.,.62,0.,(.85,.82,.75),None),
    "painted_rail":MaterialRecipe("painted_rail",1.,0.,.52,0.,(.23,.25,.26),None),
    "wood_door":MaterialRecipe("wood_door",1.,0.,.7,0.,(.34,.22,.13),None),
    "bare_handle":MaterialRecipe("bare_handle",1.,0.,.34,1.,(.66,.63,.55),None),
})


def resolve_material(profile,role):
    if profile not in ("house","apartments","office"):
        raise ValueError("unknown material profile")
    if role in ("wall","reveal"):
        return MATERIALS[{"house":"plaster_cream","apartments":"concrete_panel","office":"plaster_white"}[profile]]
    ids={"roof":"bitumen_roof","base":"stone_plinth","window":"glass_proxy",
         "balcony_door":"glass_proxy","door":"wood_door","window_frame":"painted_trim",
         "window_mullion":"painted_trim","door_frame":"wood_door","door_handle":"bare_handle",
         "balcony_slab":"concrete_panel","balcony_bar":"painted_rail",
         "balcony_panel":"painted_rail","balcony_rail":"painted_rail"}
    if role not in ids:raise ValueError("unknown material role: "+role)
    return MATERIALS[ids[role]]


def texture_maps(recipe,n=256,seed=17):
    if type(n) is not int or not 128<=n<=2048 or n&(n-1):
        raise ValueError("texture size must be a power of two in [128,2048]")
    if recipe.pattern_id is not None:
        maps,_=pattern(recipe.pattern_id,n,seed)
    else:
        maps={"basecolor":np.broadcast_to(recipe.basecolor_srgb,(n,n,3)).copy(),
              "normal":np.broadcast_to((.5,.5,1.),(n,n,3)).copy(),
              "orm":np.broadcast_to((1.,recipe.roughness,recipe.metalness),(n,n,3)).copy(),
              "height":np.zeros((n,n))}
    if any(not np.isfinite(a).all() or a.min()<0 or a.max()>1 for a in maps.values()):
        raise ValueError("invalid texture data")
    return maps


def metric_uv(vertices,repeat_m):
    """Orthonormal tangent projection in local meters; continuous on coplanar
    triangles. Vertical V is up, sloped V follows uphill, horizontal U is +X.
    Coordinates remain unwrapped. glTF export handles texture V convention.
    """
    v=np.asarray(vertices,dtype=np.float64)
    if v.shape!=(3,3) or not np.isfinite(v).all() or not math.isfinite(repeat_m) or repeat_m<=0:
        raise ValueError("invalid UV triangle or scale")
    normal=np.cross(v[1]-v[0],v[2]-v[0])
    length=np.linalg.norm(normal)
    if length<=1e-12:raise ValueError("degenerate triangle")
    normal/=length
    tangent=np.cross((0.,0.,1.),normal)
    if np.linalg.norm(tangent)<1e-10:
        tangent=np.array([1.,0.,0.])-normal*normal[0]
    tangent/=np.linalg.norm(tangent)
    bitangent=np.cross(normal,tangent)
    uv=np.column_stack((v@tangent,v@bitangent))/repeat_m
    return uv,normal
