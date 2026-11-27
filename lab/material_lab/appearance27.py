"""Deterministic appearance sidecar; no mutation of stage-26 geometry or PCGF.
All shared maps use original PCG material recipes or original analytic patterns.
Palette values are sRGB inputs, converted explicitly before shader upload.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

# Shared layers after the sixteen pre-existing material layers.
# name: source recipe, metric repeat, sRGB target, roughness, metalness.
BANK={
 'brick_red':('red_brick',1.2,(.59,.27,.19),.83,0.),
 'brick_dark':('red_brick',1.2,(.47,.22,.17),.86,0.),
 'brick_ochre':('red_brick',1.2,(.66,.35,.23),.82,0.),
 'timber_ochre':('boards',1.52,(.68,.56,.33),.72,0.),
 'timber_red':('boards',1.52,(.54,.22,.16),.75,0.),
 'timber_bluegrey':('boards_vertical',1.52,(.42,.51,.53),.73,0.),
 'stucco_warm':('plaster_cream',2.,(.76,.67,.53),.83,0.),
 'stucco_white':('plaster_white',2.,(.86,.85,.78),.81,0.),
 'stucco_stone':('plaster_cream',2.,(.62,.60,.54),.84,0.),
 'panel_gray':('concrete_panel',2.4,(.60,.60,.57),.82,0.),
 'panel_warm':('concrete_panel',2.4,(.69,.66,.58),.83,0.),
 'panel_white':('concrete_panel',2.4,(.77,.79,.78),.72,0.),
 'metal_red':('standing_seam',2.4,(.43,.18,.14),.46,.06),
 'metal_graphite':('standing_seam',2.4,(.26,.29,.30),.45,.06),
 'metal_green':('standing_seam',2.4,(.30,.38,.32),.47,.06),
 'clay_red':('ceramic_roof',1.2,(.63,.32,.20),.77,0.),
 'clay_aged':('ceramic_roof',1.2,(.46,.24,.18),.86,0.),
 'membrane_gray':('bitumen_roof',2.,(.28,.29,.27),.88,0.),
 'membrane_dark':('bitumen_roof',2.,(.20,.22,.22),.89,0.),
 'stone_sand':('stone_plinth',1.6,(.59,.55,.47),.85,0.),
}
LAYERS={name:(16+i,x[1]) for i,(name,x) in enumerate(BANK.items())}
assert [LAYERS[k][0] for k in ('membrane_gray','membrane_dark','stone_sand')]==[33,34,35]
# Body, roof family, window frame, balcony panel, entry, age, rail pattern.
# No claim of observed regional distributions: illustrative authored palettes.
PROFILES={
 'baltic_timber':(('timber_ochre','timber_bluegrey'),('metal_red','metal_graphite'),(.82,.83,.76),(.65,.62,.48),(.26,.32,.26),(.30,.65),(3,5)),
 'nordic_cottage':(('timber_red','timber_bluegrey'),('metal_graphite','metal_green'),(.90,.89,.82),(.78,.78,.70),(.35,.25,.19),(.15,.43),(3,4)),
 'central_townhouse':(('brick_red','brick_dark'),('clay_red','clay_aged'),(.32,.23,.17),(.27,.30,.28),(.26,.20,.14),(.40,.75),(3,5)),
 'mediterranean_villa':(('stucco_white','stucco_warm'),('clay_red','clay_aged'),(.30,.37,.31),(.76,.73,.60),(.22,.31,.25),(.20,.55),(3,5)),
 'caucasus_house':(('stucco_stone','stucco_warm'),('metal_red','metal_green'),(.38,.29,.22),(.64,.55,.40),(.30,.24,.16),(.30,.75),(3,5)),
 'panel_slab':(('panel_gray','panel_warm'),('membrane_gray','membrane_dark'),(.84,.85,.80),(.52,.56,.48),(.22,.34,.29),(.50,.85),(1,3)),
 'brick_walkup':(('brick_red','brick_ochre'),('clay_aged','clay_red'),(.72,.68,.57),(.46,.51,.42),(.28,.34,.25),(.45,.78),(3,5)),
 'perimeter_block':(('stucco_warm','stucco_stone'),('clay_red','clay_aged'),(.78,.78,.70),(.63,.60,.48),(.30,.26,.20),(.30,.72),(3,5)),
 'mediterranean_balcony':(('stucco_white','stucco_warm'),('membrane_gray','membrane_dark'),(.88,.88,.82),(.79,.76,.62),(.31,.38,.30),(.18,.50),(3,5)),
 'caucasus_tower':(('panel_warm','panel_gray'),('membrane_dark','membrane_gray'),(.74,.73,.66),(.43,.50,.53),(.31,.32,.28),(.32,.72),(1,4)),
 'modern_gallery':(('stucco_white','panel_white'),('membrane_dark','membrane_gray'),(.29,.32,.32),(.45,.49,.50),(.23,.27,.29),(.08,.30),(4,3)),
 'masonry_office':(('brick_dark','brick_red'),('clay_aged','clay_red'),(.28,.30,.30),(.38,.42,.43),(.23,.25,.26),(.25,.55),(4,3)),
 'modern_ribbon':(('panel_white','panel_gray'),('membrane_dark','membrane_gray'),(.24,.28,.29),(.42,.47,.49),(.23,.27,.30),(.05,.26),(4,3)),
 'glass_grid':(('panel_white','panel_gray'),('membrane_gray','membrane_dark'),(.21,.26,.29),(.40,.46,.49),(.24,.30,.33),(.04,.22),(4,3)),
 'civic_block':(('stone_sand',),('clay_aged','clay_red'),(.36,.34,.29),(.55,.53,.46),(.29,.30,.27),(.25,.57),(5,3)),
}
GLASS=((.043,.76,.115),(.055,.60,.145),(.115,.36,.105)) # effective F0, transmission, roughness


def stable(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')


def linear(a):
    a=np.asarray(a,np.float32);return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)


def srgb(a):
    a=np.clip(a,0.,1.);return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)


def glass_weights(cosine,f0,trans):
    """Two parallel incoherent dielectric interfaces, effective coating F0.
    Reflection + transmission <= 1; coating/absorption removes remaining energy.
    Mirrors the GLSL implementation, not a spectral multilayer optical solver.
    """
    f=f0+(1-f0)*(1-np.clip(cosine,0.,1.))**5
    r=2*f/(1+f);return np.stack((r,(1-r)*trans),axis=-1)


def profile(record):
    a=record['decision']['archetype'];name=a['id'];bid=record['footprint']['building_id']
    body,roof,frame,panel,entry,age,rail=PROFILES[name];k=stable('appearance27',bid)
    use={'house':0,'apartments':1,'office':2}[a['use']]
    g=2 if use==2 else 1 if name in ('modern_gallery','caucasus_tower') else 0
    choose=lambda seq,domain:seq[stable(domain,bid)%len(seq)]
    roof_name=choose(roof,'roof') if record['decision']['actual_roof']!='flat' else choose(('membrane_gray','membrane_dark'),'membrane')
    tint=.94+.12*((k>>8)%100)/99
    return dict(building=bid,archetype=name,use=use,glass=g,body=choose(body,'body'),roof=roof_name,
        frame=list(frame),panel=list(panel),entry=list(entry),age=age[0]+(age[1]-age[0])*((k>>16)%101)/100,
        rail=choose(rail,'rail'),low=choose((.28,.32,.36),'lower'),entry_style=0 if use==0 else 2 if use==2 or name=='modern_gallery' else 1,
        entry_variant=int((k>>24)%3),tint=tint,seed=int(k%4096),origin=list(record['offset']))


def periodic_noise(x,y,key):
    rng=np.random.default_rng(key);q=np.zeros_like(x)
    for i,amp in zip((2,4,9,19,43),(.33,.25,.19,.14,.09)):
        a,b=rng.uniform(0,2*np.pi,2);q+=amp*np.sin(2*np.pi*(i*x+(i-1)*y)+a)*np.cos(2*np.pi*((i+1)*x-i*y)+b)
    return q


def normal_from_height(h,repeat):
    dx=(np.roll(h,-1,1)-np.roll(h,1,1))*len(h)/(2*repeat)
    dy=(np.roll(h,-1,0)-np.roll(h,1,0))*len(h)/(2*repeat)
    n=np.stack((-dx,dy,np.ones_like(h)),axis=-1);n/=np.linalg.norm(n,axis=-1,keepdims=True)
    return n*.5+.5


def material(name,root,n=512):
    source,repeat,color,rough,metal=BANK[name];key=stable('material27',name);x,y=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n);noise=periodic_noise(x,y,key)
    rng=np.random.default_rng(key);grain=rng.uniform(-1,1,(n,n))*.012
    target=np.asarray(color);read=lambda kind:np.asarray(Image.open(root/source/(kind+'.png')).convert('RGB').resize((n,n),Image.Resampling.LANCZOS),np.float32)/255.
    if source.startswith('boards'):
        u,v=(y,x) if source.endswith('vertical') else (x,y)
        z=v*8;d=np.minimum(z%1,1-z%1);groove=np.exp(-((d/.035)**2));board=np.floor(z).astype(int)
        boardtone=np.random.default_rng(key+1).uniform(-.085,.085,8)[board]
        fiber=np.sin(2*np.pi*(v*92+.60*np.sin(u*2*np.pi)+noise*.45))*.012
        h=-.003*groove+.00032*noise+.00016*fiber/.012
        a=target*(1+boardtone[...,None]+.12*noise[...,None]+fiber[...,None]-.24*groove[...,None])+grain[...,None]
        normal=normal_from_height(h,repeat);occ=1-.27*groove
    elif source=='standing_seam':
        d=np.minimum((x*5)%1,1-(x*5)%1);rib=np.exp(-((d/.027)**2));h=.009*rib+.00045*noise
        a=target*(1+.10*noise[...,None]-.13*rib[...,None])+grain[...,None]
        normal=normal_from_height(h,repeat);occ=1-.12*rib
    else:
        old=read('basecolor');normal=read('normal');orm=read('orm');occ=orm[:,:,0]
        lum=old@np.array([.2126,.7152,.0722]);detail=lum/max(float(np.median(lum)),.1)-1.
        if source=='red_brick':
            mask=np.clip((old[:,:,0]-old[:,:,1]-.055)/.13,0,1)
            brick=target*(1+.46*detail[...,None]+.18*noise[...,None])+grain[...,None]
            mortar=np.array([.47,.46,.42])*(1+.18*detail[...,None]+.09*noise[...,None])+grain[...,None]*.6
            a=mortar*(1-mask[...,None])+brick*mask[...,None]
            assert float(mask.mean())>.50 and float(a[:,:,0][mask>.8].mean()-a[:,:,1][mask>.8].mean())>.14
        else:
            strength=.32 if source.startswith('plaster') or source=='concrete_panel' else .74
            a=target*(1+strength*detail[...,None]+.13*noise[...,None])+grain[...,None]
    r=np.clip(rough+.035*noise+.018*grain/.012,.10,.98);orm=np.stack((np.clip(occ,0,1),r,np.full_like(r,metal)),axis=-1)
    a=np.clip(a,0,1);normal=np.clip(normal,0,1);result=(a.astype(np.float32),normal.astype(np.float32),orm.astype(np.float32))
    for m in result:assert m.shape==(n,n,3) and np.isfinite(m).all() and m.min()>=0 and m.max()<=1
    return result


def add_materials(arrays,root,out):
    """Shared layer IDs fixed by BANK; export the actual quantized maps used.
    Existing sources remain available through the stage-26 material loader.
    """
    layers=[[],[],[]];manifest=[];directory=out/'material_bank';directory.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(5*240,4*204),(238,239,240));d=ImageDraw.Draw(sheet)
    for i,name in enumerate(BANK):
        p=directory/name;p.mkdir(exist_ok=True);maps=material(name,root)
        for j,(key,m) in enumerate(zip(('basecolor','normal','orm'),maps)):
            q=np.rint(m*255).astype(np.uint8);Image.fromarray(q).save(p/(key+'.png'));m=q.astype(np.float32)/255
            layers[j].append(np.dstack((linear(m),np.ones(m.shape[:2],np.float32))) if j==0 else m)
        im=Image.open(p/'basecolor.png').resize((224,176));x=i%5*240;y=i//5*204;sheet.paste(im,(x+8,y));d.text((x+8,y+178),name,fill='black')
        manifest.append(dict(id=16+i,name=name,source=BANK[name][0],repeat_m=BANK[name][1],colorspace=['sRGB','linear','linear'],roughness=BANK[name][3],metalness=BANK[name][4]))
    sheet.save(out/'material_contact.png')
    for j in range(3):arrays[j]=np.concatenate((arrays[j],np.asarray(layers[j],np.float32)),axis=0)
    assert len(arrays[0])==len(arrays[1])==len(arrays[2])==16+len(BANK)
    return manifest


def railing(kind,n=1024):
    """Original metric alpha masks on the old 0.48 m repeat, no extra faces.
    3 vertical pickets; 4 horizontal rails; 5 decorative diagonal cross.
    All include top/bottom rails and remain genuinely open.
    """
    s=n*2;x,y=np.meshgrid((np.arange(s)+.5)/s,(np.arange(s)+.5)/s)
    dist=lambda t:np.abs((t+.5)%1-.5)
    edge=(y<.036)|(y>.964)
    if kind==3:m=edge|(dist(3*x)<.055)
    elif kind==4:m=edge|(dist(x)<.035)|(dist(3*y)<.044)
    elif kind==5:m=edge|(dist(2*x)<.043)|(dist(x+2*y)<.025)|(dist(x-2*y)<.025)
    else:raise ValueError(kind)
    a=m.reshape(n,2,n,2).mean(axis=(1,3)).astype(np.float32)
    assert .12<float((a>=.5).mean())<.40
    rgba=np.ones((n,n,4),np.float32);rgba[:,:,:3]=linear([.56,.55,.52]);rgba[:,:,3]=a
    return rgba


def mask_mips(base):
    """Unscaled box-filter pyramid. Correct alpha coverage at cutoff .5 per
    railing layer; original antenna atlas is excluded from global scaling.
    """
    raw=np.array(base,np.float32);n=len(raw[0]);assert n==raw.shape[2] and n&(n-1)==0
    target=(raw[:,:,:,3]>=.5).mean(axis=(1,2));levels=[raw.copy()];report=[]
    while n>1:
        n//=2;raw=raw.reshape(len(raw),n,2,n,2,4).mean(axis=(2,4));q=raw.copy();scales=[];errors=[];uncorrected=[]
        for j in range(len(q)):
            a=raw[j,:,:,3];k=1.;uncorrected.append(float(abs((a>=.5).mean()-target[j])))
            if j!=2:
                lo,hi=0.,16.
                for _ in range(24):
                    mid=(lo+hi)/2
                    if (a*mid>=.5).mean()<target[j]:lo=mid
                    else:hi=mid
                k=min((1.,lo,hi),key=lambda x:(abs(float((a*x>=.5).mean())-target[j]),abs(x-1)))
            q[j,:,:,3]=np.clip(a*k,0,1);scales.append(float(k));errors.append(float(abs((q[j,:,:,3]>=.5).mean()-target[j])))
        assert all(e<=u+1e-12 for e,u in zip(errors,uncorrected))
        levels.append(q);report.append(dict(size=n,scales=scales,coverage_error=errors,uncorrected_error=uncorrected))
    return levels,dict(base_coverage=target.tolist(),levels=report,antenna_unscaled=True)


def accessory_code(r):
    k=r.get('accessory','')
    if not k:return 1 if r['role']=='roof' else 9 if r['role']=='base' else 0
    if k in ('eave_cover','dormer_roof'):return 2
    if k=='canopy':return 3
    if k in ('slab','stage21_cap'):return 4
    if k in ('plinth','belt','soffit','dormer_trim'):return 5
    if k in ('parapet','coping'):return 7
    if k in ('gutter_side','gutter_floor','bent_drain','flashing','flue'):return 8
    if k in ('stage28_entry_cap','stage28_balcony_edge','stage28_sill'):return {'stage28_entry_cap':10,'stage28_balcony_edge':11,'stage28_sill':12}[k]
    return 6


def augment(meta,table,profiles,cutout_pairs):
    """Original record and opening indices are retained. Extra appearance
    columns are separate from PCGF. Cutout records are appended by building
    and kind, not duplicated per pixel or per individual balcony.
    """
    m=np.zeros((len(meta)+len(cutout_pairs),14,4),np.float32);m[:len(meta),:8]=meta
    def fill(j,p,code):
        m[j,8]=[p['use'],p['glass'],p['age'],p['seed']];m[j,9]=[*linear(p['frame']),GLASS[p['glass']][2]]
        m[j,10]=[p['tint'],p['entry_style'],p['entry_variant'],0.];m[j,11]=[*linear(p['panel']),p['low']]
        m[j,12]=[p['origin'][0],p['origin'][1],code,p['rail']];m[j,13]=[*linear(p['entry']),0.]
    for r in table:
        j=r['id'];p=profiles[r['building']];code=accessory_code(r);fill(j,p,code)
        k=r.get('accessory','');fam=r['family'];name=None
        if r['role']=='wall':name=p['body']
        elif fam=='red_brick':name=p['body'] if p['body'].startswith('brick') else 'brick_red'
        elif fam=='ceramic_roof':name=p['roof'] if p['roof'].startswith('clay') else 'clay_red'
        elif fam=='metal_roof_red':name=p['roof'] if p['roof'].startswith('metal') else 'metal_graphite'
        elif fam=='bitumen_roof':name=p['roof'] if p['roof'].startswith('membrane') else 'membrane_dark'
        elif fam=='concrete_panel':name=p['body'] if p['body'].startswith('panel') else 'panel_gray'
        elif fam in ('plaster_white','plaster_cream'):name=p['body'] if p['body'].startswith('stucco') else 'stucco_warm'
        elif fam=='stone_plinth':name='stone_sand'
        if name is not None and k!='vent':m[j,0,1:3]=LAYERS[name]
        if m[j,0,0]==2:m[j,1,1]=p['low']
    remap={}
    for i,(bid,kind) in enumerate(sorted(cutout_pairs)):
        j=len(meta)+i;p=profiles[bid];layer=0 if kind=='roof' else 2 if kind=='antenna' else p['rail'];m[j,0]=[3,layer,1,p['seed']]
        fill(j,p,0);remap[bid,kind]=j
    assert np.isfinite(m).all() and max(remap.values())<65536
    np.testing.assert_array_equal(m[:len(meta),0,0],meta[:,0,0]);np.testing.assert_array_equal(m[:len(meta),2:8],meta[:,2:8])
    return m,remap
