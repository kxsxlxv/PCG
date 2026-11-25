"""Original PCG research 06 pattern kernel; promoted without algorithm changes.
Source research/06_materials/material_probe.py at main 3a8ac40ce8b498bea974bfaeaaca9b2dd53e6928.
Synthetic diagnostic patterns, not measured regional materials.
"""
import hashlib
import numpy as np

RECIPES = {
    'plaster_cream': ('plaster', (194,180,155), 2.0, .003, .84, 0),
    'plaster_white': ('plaster', (215,216,207), 2.0, .002, .79, 0),
    'red_brick': ('brick', (151,77,57), 1.2, .009, .86, 0),
    'concrete_panel': ('concrete', (157,155,146), 2.4, .004, .83, 0),
    'stone_plinth': ('stone', (130,121,107), 1.6, .014, .87, 0),
    'metal_roof_red': ('metal', (133,62,49), 2.4, .018, .52, 0),
    'ceramic_roof': ('tile', (153,82,56), 1.2, .018, .76, 0),
    'bitumen_roof': ('bitumen', (82,84,80), 2.0, .003, .94, 0),
}
def clamp(x): return np.clip(x,0,1)
def stable(*parts): return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def noise(n,k,rng):
    # Periodic value noise, texel centers. Wrap both lattice coordinates.
    a=rng.random((k,k)); t=(np.arange(n)+.5)*k/n; i=np.floor(t).astype(int); f=t-i; f=f*f*(3-2*f)
    lo=a[np.ix_(i%k,i%k)]; x=a[np.ix_(i%k,(i+1)%k)]; y=a[np.ix_((i+1)%k,i%k)]; xy=a[np.ix_((i+1)%k,(i+1)%k)]
    return (lo*(1-f)+x*f)*(1-f[:,None])+(y*(1-f)+xy*f)*f[:,None]
def material(name,n,seed=17):
    kind,color,size,amplitude,rough,metal=RECIPES[name]; rng=np.random.default_rng(stable(seed,name))
    u,v=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n)
    macro=noise(n,4,rng)-.5; mid=noise(n,16,rng)-.5; fine=noise(n,min(128,n),rng)-.5
    gran=rng.random((n,n))-.5; tex=.48*mid+.30*fine+.22*gran
    h=.5+.32*tex; c=np.array(color)/255*(1+.18*macro[...,None]+.13*tex[...,None]); ao=np.ones((n,n)); r=rough+.10*tex
    if kind in ('brick','stone','tile'):
        cols,rows={'brick':(5,16),'stone':(4,6),'tile':(4,5)}[kind]
        row=np.floor(v*rows).astype(int); xx=(u*cols+.5*(row%2))%1; yy=(v*rows)%1
        # Stagger requires even row count for seamless repeat; odd tile rows stay aligned.
        if rows%2: xx=(u*cols)%1
        col=np.floor(u*cols+(0 if rows%2 else .5*(row%2))).astype(int)%cols
        cell=rng.random((rows,cols))[row%rows,col]-.5
        edge=np.minimum(np.minimum(xx,1-xx),np.minimum(yy,1-yy)); mortar=clamp((edge-.035)/.05)
        if kind=='tile':
            h=.22+.60*mortar*(.55+.45*np.sin(np.pi*xx)**2)+.08*tex
            c=c*(1+.28*cell[...,None]); c=c*(.77+.23*mortar[...,None]); ao=.62+.38*mortar
        else:
            h=.18+.68*mortar+.12*tex; joint=np.array([.45,.43,.39]) if kind=='brick' else np.array([.34,.32,.29])
            c=(c*(1+.32*cell[...,None]))*mortar[...,None]+joint*(1-mortar[...,None]); ao=.72+.28*mortar
    elif kind=='metal':
        ribs=(.5+.5*np.cos(2*np.pi*u*8))**8; seam=(.5+.5*np.cos(2*np.pi*v*4))**24
        h=.24+.62*ribs+.05*seam+.035*tex; c=c*(.94+.06*tex[...,None]); ao=.92-.07*ribs; r=.52+.06*macro
    elif kind=='concrete':
        pores=clamp((gran-.40)*10); h=.50+.19*tex-.18*pores; c=c*(1-.16*pores[...,None]); ao=1-.22*pores
    elif kind=='bitumen':
        seam=(.5+.5*np.cos(2*np.pi*u*4))**45; h=.45+.30*tex+.17*seam; c=c*(1-.12*seam[...,None]+.27*gran[...,None]); ao=1-.08*seam
    h=clamp(h); step=size/n
    dx=(np.roll(h,-1,1)-np.roll(h,1,1))*amplitude/(2*step)
    dy=(np.roll(h,-1,0)-np.roll(h,1,0))*amplitude/(2*step)
    normal=np.stack([-dx,dy,np.ones_like(h)],-1); normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    orm=np.stack([clamp(ao),clamp(r),np.full_like(h,metal)],-1)
    return {'basecolor':clamp(c),'normal':normal*.5+.5,'orm':orm,'height':h},normal

