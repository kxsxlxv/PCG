"""Facade-wide linear emission masks from canonical openings.

Reuses the original research/06_materials/material_probe.py linear box mip
concept. Fractional rectangle/texel coverage avoids resolution-dependent
energy loss. Aperture emission intent only: no frame/rail occlusion or lighting.
"""
import math
import numpy as np


def facade_emission(facade, height, intents, size=128):
    if type(size) is not int or not 2<=size<=1024 or size&(size-1):
        raise ValueError("size must be a power of two in [2,1024]")
    if not math.isfinite(height) or height<=0 or not math.isfinite(facade.length) or facade.length<=0:
        raise ValueError("invalid facade dimensions")
    by_id={i.opening_id:i for i in intents}
    if len(by_id)!=len(intents):
        raise ValueError("duplicate occupancy intent")
    xe=np.linspace(0,facade.length,size+1)
    ze=np.linspace(0,height,size+1)
    dx=facade.length/size; dz=height/size
    mask=np.zeros((size,size,3),dtype=np.float64)
    coverage=np.zeros((size,size),dtype=np.float64)
    integral=np.zeros(3)
    for o in sorted(facade.openings,key=lambda o:o.id):
        if o.role not in ("window","balcony_door"):
            continue
        if o.id not in by_id:
            raise ValueError("missing glazing intent")
        if not (0<=o.u<o.u+o.width<=facade.length and 0<=o.z<o.z+o.height<=height):
            raise ValueError("glazing outside facade")
        intent=by_id[o.id]
        if not math.isclose(intent.area_m2,o.width*o.height,rel_tol=1e-12):
            raise ValueError("glazing intent area mismatch")
        x=np.maximum(0,np.minimum(xe[1:],o.u+o.width)-np.maximum(xe[:-1],o.u))/dx
        z=np.maximum(0,np.minimum(ze[1:],o.z+o.height)-np.maximum(ze[:-1],o.z))/dz
        fraction=z[::-1,None]*x[None,:]  # image row zero is facade top
        coverage+=fraction
        emission=np.asarray(intent.emissive_linear,dtype=float)
        mask+=fraction[...,None]*emission
        integral+=o.width*o.height*emission
    if coverage.max()>1+1e-10:
        raise ValueError("overlapping glazing coverage")
    np.testing.assert_allclose(mask.mean((0,1)),integral/(facade.length*height),atol=1e-12,rtol=0)
    return mask


def linear_mips(mask):
    """Unquantized linear box chain, each level preserves the facade mean."""
    a=np.asarray(mask,dtype=np.float64)
    if a.ndim!=3 or a.shape[2]!=3 or a.shape[0]!=a.shape[1]:
        raise ValueError("expected square RGB mask")
    n=a.shape[0]
    if n<1 or n&(n-1) or not np.isfinite(a).all() or a.min()<0 or a.max()>1+1e-10:
        raise ValueError("invalid linear mask")
    levels=[a.copy()]
    mean=a.mean((0,1))
    while n>1:
        a=a.reshape(n//2,2,n//2,2,3).mean((1,3))
        np.testing.assert_allclose(a.mean((0,1)),mean,atol=1e-12,rtol=0)
        levels.append(a);n//=2
    return tuple(levels)


def srgb_encode(a):
    a=np.clip(a,0,1)
    return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)


def srgb_decode(a):
    return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
