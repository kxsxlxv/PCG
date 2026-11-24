"""Original procedural PBR research probe. Python 3, NumPy and Pillow."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

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

def bytes_bc(n,block):
    total=0
    while True:
        total+=max(1,math.ceil(n/4))**2*block
        if n==1:return total
        n=max(1,n//2)
def encode(a):return np.round(clamp(a)*255).astype('uint8')
def rgb(a):return Image.fromarray(encode(a)).convert('RGB')
def srgb_to_linear(a):return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
def linear_to_srgb(a):a=clamp(a);return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)
def facade(out,seed=17):
    # Illustrative orthographic facade, not a PBR engine render or real building.
    w,h=960,384; base=np.array([.76,.70,.59]); y,x=np.mgrid[:h,:w]; noise_map=noise(h,12,np.random.default_rng(seed))
    noise_map=np.tile(noise_map,(1,math.ceil(w/h)))[:,:w]
    day=clamp(np.broadcast_to(base,(h,w,3)).copy()*(.91+.15*noise_map[...,None]))
    emiss=np.zeros_like(day); slots=[]
    for floor in range(4):
        for bay in range(12):
            x0=bay*80+19;y0=floor*96+21;x1=x0+42;y1=y0+54
            day[y0-4:y1+4,x0-4:x1+4]=[.40,.37,.31];day[y0:y1,x0:x1]=[.14,.23,.29]
            day[y0:y1,x0:x0+3]=[.85,.83,.78];day[y0:y1,x1-3:x1]=[.85,.83,.78];day[y0:y0+3,x0:x1]=[.85,.83,.78];day[y1-3:y1,x0:x1]=[.85,.83,.78];day[y0:y1,x0+20:x0+23]=[.81,.80,.75]
            rng=np.random.default_rng(stable(seed,'building_42',floor,bay,'occupancy')); lit=bool(rng.random()<.32)
            if lit:
                power=.45+.50*rng.random(); hue=np.array([1,.66+.18*rng.random(),.35+.28*rng.random()]); emiss[y0+3:y1-3,x0+3:x1-3]=power*hue;emiss[y0+3:y1-3,x0+20:x0+23]=0
            slots.append({'floor':floor,'bay':bay,'lit':lit})
    night=linear_to_srgb(srgb_to_linear(day)*.055+emiss)
    rgb(day).save(out/'facade_day.png');rgb(night).save(out/'facade_night.png');rgb(linear_to_srgb(emiss)).save(out/'facade_emissive_display.png')
    # Linear box mip: mean radiance preserved exactly for divisible dimensions.
    mip=emiss.reshape(h//8,8,w//8,8,3).mean((1,3));err=float(np.max(np.abs(mip.mean((0,1))-emiss.mean((0,1)))))
    assert err<1e-12
    return {'windows':len(slots),'lit':sum(s['lit'] for s in slots),'mean_linear_emissive':emiss.mean((0,1)).tolist(),'mip_mean_error':err,'slots':slots}
def run(out,n):
    out.mkdir(parents=True,exist_ok=True);stats={}; sheet=Image.new('RGB',(840,8*176+36),(35,38,43)); draw=ImageDraw.Draw(sheet)
    for i,label in enumerate(['Base color','Normal','ORM','Height','Diffuse preview']):draw.text((i*168+7,9),label,fill='white')
    for i,name in enumerate(RECIPES):
        maps,normals=material(name,n); again,_=material(name,n)
        assert all(np.array_equal(maps[k],again[k]) for k in maps)
        d=out/name;d.mkdir(exist_ok=True);record={}
        for k,a in maps.items():
            assert np.isfinite(a).all() and a.min()>=-1e-9 and a.max()<=1+1e-9
            p=d/(k+'.png');Image.fromarray(encode(a)).save(p)
            assert np.array_equal(np.asarray(Image.open(p)),encode(a))
            record[k]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        nl=float(np.max(np.abs(np.linalg.norm(normals,axis=-1)-1)));assert nl<1e-12
        # Diagnostic: seam step relative to mean interior neighbor step, not proof of seamlessness.
        a=maps['height'];ratio=[]
        for axis in (0,1):
            interior=np.mean(np.abs(np.diff(a,axis=axis))); seam=np.mean(np.abs(np.take(a,0,axis=axis)-np.take(a,-1,axis=axis)));ratio.append(float(seam/max(interior,1e-12)))
        stats[name]={'tile_m':RECIPES[name][2],'height_amplitude_m':RECIPES[name][3],'normal_max_length_error':nl,'height_seam_step_ratio_yx':ratio,'maps':record}
        l=np.array([-.46,.40,.79]);l/=np.linalg.norm(l);shade=.34+.66*np.maximum(0,normals@l);preview=linear_to_srgb(srgb_to_linear(maps['basecolor'])*shade[...,None]*maps['orm'][...,0,None]);
        for j,a in enumerate([*maps.values(),preview]):sheet.paste(rgb(a).resize((156,156),Image.Resampling.LANCZOS),(j*168+6,36+i*176))
        draw.text((7,194+i*176),name,fill='white')
    sheet.save(out/'material_contact.png')
    f=facade(out); assert f==facade(out)
    memory=[{'size':s,'layers':32,'three_maps_mib':32*3*bytes_bc(s,16)/2**20,'with_height_mib':32*(3*bytes_bc(s,16)+bytes_bc(s,8))/2**20} for s in (512,1024,2048,4096)]
    result={'resolution':n,'materials':stats,'maps_written':len(stats)*4,'facade':f,'memory_model':memory,'limitations':['synthetic patterns, no photographic calibration','diffuse preview is not PBR rendering','height seam ratios are diagnostic only','no compressed textures or engine tests','no material maker execution']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='generated_materials');p.add_argument('--size',type=int,default=512);a=p.parse_args()
    if a.size<128 or a.size>4096 or a.size&(a.size-1):p.error('size must be a power of two in [128,4096]')
    r=run(Path(a.output),a.size);print(json.dumps({'materials':len(r['materials']),'maps':r['maps_written'],'lit':r['facade']['lit'],'windows':r['facade']['windows'],'mip_error':r['facade']['mip_mean_error'],'memory':r['memory_model']},indent=2))
