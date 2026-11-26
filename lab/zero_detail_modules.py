"""Research-only original shared window/door relief maps; no external assets."""
import hashlib
import json
import numpy as np
from PIL import Image
from pcg._material_patterns import noise
from pcg.far_emission import srgb_decode

STYLES=("white_casement","cream_cross","dark_triple","white_transom",
        "wood_double","dark_modern","cream_shutter","gray_office",
        "balcony_white","balcony_dark","timber_entry","metal_entry")

def generate(n=256):
    x,y=np.meshgrid((np.arange(n)+.5)/n,1-(np.arange(n)+.5)/n)
    colors=[];normals=[];masks=[]
    def smooth(a,lo,hi):
        t=np.clip((a-lo)/(hi-lo),0,1);return t*t*(3-2*t)
    for i,name in enumerate(STYLES):
        rng=np.random.default_rng(9103+i)
        grain=noise(n,32,rng)-.5;fine=noise(n,96,rng)-.5
        edge=np.minimum.reduce([x,1-x,y,1-y])
        outer=.105 if i<8 else .085
        inner=outer+.045
        reveal=1-smooth(edge,outer-.025,outer+.005)
        frame=1-smooth(edge,inner-.01,inner+.008)
        mull=np.zeros_like(x)
        bars={0:(.5,),1:(.5,),2:(1/3,2/3),3:(.5,),4:(.5,),5:(),6:(.5,),7:(.5,),8:(.5,),9:(.35,)}.get(i,())
        for c in bars:mull=np.maximum(mull,1-smooth(np.abs(x-c),.018,.030))
        if i in (1,3,7,8):mull=np.maximum(mull,1-smooth(np.abs(y-(.72 if i!=8 else .32)),.016,.027))
        sill=(1-smooth(y,.045,.075))*smooth(x,.015,.06)*smooth(1-x,.015,.06)
        frame=np.maximum(frame,mull)
        glass=(1-frame)*(1-sill) if i<10 else np.zeros_like(x)
        palettes=[(.84,.84,.80),(.78,.75,.66),(.27,.29,.30),(.87,.87,.84),
                  (.43,.28,.17),(.23,.25,.27),(.73,.71,.63),(.45,.47,.48),
                  (.83,.84,.80),(.25,.27,.29),(.40,.25,.15),(.33,.35,.36)]
        paint=np.asarray(palettes[i])
        rgb=np.broadcast_to(paint,(n,n,3)).copy()*(.58+.42*smooth(edge,0,outer))[...,None]
        rgb*=1+(.07*grain+.035*fine)[...,None]
        rgb=rgb*(1-sill[...,None])+np.array([.69,.66,.59])*sill[...,None]
        curtain=np.clip(.47+.45*smooth(np.abs(x-.5),.10,.37)*(.78+.12*np.sin(x*80+i)+.06*grain),0,1)
        tint=np.stack((.13+.10*y,.20+.11*y,.25+.14*y),-1)*(.82+.18*curtain[...,None])
        rgb=rgb*(1-glass[...,None])+tint*glass[...,None]
        height=.16+.46*frame+.27*reveal+.26*sill+.015*fine
        if i>=10:
            panel=(smooth(x,.16,.20)*smooth(1-x,.16,.20)*
                   np.maximum(smooth(y,.15,.19)*smooth(.46-y,0,.04),
                              smooth(y,.55,.59)*smooth(.86-y,0,.04)))
            height=.38+.20*reveal+.21*panel+.02*grain
            rgb*=1-.14*panel[...,None]
            handle=(np.abs(x-.79)<.035)&(np.abs(y-.49)<.012)
            lock=(np.abs(x-.79)<.013)&(np.abs(y-.45)<.013)
            rgb[handle|lock]=[.69,.62,.43];height[handle|lock]=.95
        height=np.clip(height,0,1)
        gy,gx=np.gradient(height,1/n,1/n)
        normal=np.stack((-gx*.018/1.4,gy*.018/1.5,np.ones_like(x)),axis=-1)
        normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
        nh=np.concatenate((normal*.5+.5,height[...,None]),axis=-1)
        ao=np.clip(.66+.34*smooth(edge,0,inner)-.12*reveal,0,1)
        mask=np.stack((glass,curtain,.66*(1-glass)+.16*glass,ao),-1)
        colors.append(np.clip(rgb,0,1));normals.append(nh);masks.append(mask)
    return tuple(np.asarray(a,dtype=np.float32) for a in (colors,normals,masks))

def library(out,n=256):
    arrays=generate(n)
    for a,b in zip(arrays,generate(n)):
        np.testing.assert_array_equal(a,b)
        assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    color,nh,mask=arrays
    assert np.all(mask[10:,:,:,0]==0)
    assert np.all((mask[:10,:,:,0].mean((1,2))>.25)&(mask[:10,:,:,0].mean((1,2))<.85))
    assert np.all(mask[:10,0,:,0]==0) and np.all(mask[:10,-1,:,0]==0)
    np.testing.assert_allclose(np.linalg.norm(nh[...,:3]*2-1,axis=-1),1,atol=2e-6)
    entries=[]
    for i,name in enumerate(STYLES):
        d=out/name;d.mkdir(parents=True,exist_ok=True);files={}
        for key,a in (("basecolor",color),("normal_height",nh),("mask",mask)):
            p=d/(key+".png");Image.fromarray(np.uint8(np.round(a[i]*255))).save(p)
            files[key]={"path":str(p.relative_to(out)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
        entries.append({"style":i,"name":name,"files":files,"glass_coverage":float(mask[i,:,:,0].mean())})
    (out/"manifest.json").write_text(json.dumps({"schema":"pcg.research.opening-modules.v1",
        "resolution":n,"styles":entries,"provenance":"Original synthetic authored masks; not photographs",
        "channels":{"basecolor":"sRGB","normal_height":"linear tangent normal RGB + height A",
        "mask":"linear glazing/curtain/roughness/AO"}},indent=2))
    return srgb_decode(color),nh,mask
