"""Original isotropic shared microtextures and tile-local reflection proxies.
The authored palette, texture dimensions, layer IDs and alpha banks are fixed.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
import appearance27 as ap

UNSTRUCTURED=('stucco_warm','stucco_white','stucco_stone','panel_gray','panel_warm','panel_white')


def field(n,rng,low,high):
    # Isotropic band-limited periodic random field. Unlike the stage-27
    # sinusoidal sum, energy is spread across directions/frequencies.
    ky=np.fft.fftfreq(n)*n;kx=np.fft.rfftfreq(n)*n;f2=ky[:,None]**2+kx[None,:]**2
    w=(1-np.exp(-f2/(2*low**2)))*np.exp(-f2/(2*high**2))
    z=np.fft.irfft2(np.fft.rfft2(rng.standard_normal((n,n)))*w,s=(n,n),axes=(0,1));z-=z.mean();z/=z.std()
    return np.clip(z,-3.5,3.5)


def material(name,n=512):
    src,repeat,color,rough,metal=ap.BANK[name];rng=np.random.default_rng(ap.stable('isotropic28',name))
    coarse=field(n,rng,1.4,8.);mid=field(n,rng,7.,30.);fine=field(n,rng,29.,114.);grain=rng.standard_normal((n,n))
    concrete=src=='concrete_panel';pores=(grain>2.65).astype(float) if concrete else np.zeros_like(grain)
    tone=1.+.022*coarse+.012*mid+.007*fine-.045*pores
    albedo=np.asarray(color)[None,None,:]*tone[:,:,None]+.002*grain[:,:,None]
    h=.00055*mid+.00012*fine-.00045*pores if concrete else .00048*mid+.00015*fine
    normal=ap.normal_from_height(h,repeat);orm=np.stack((1.-.18*pores,np.clip(rough+.019*mid+.011*fine,.1,.98),np.full((n,n),metal)),axis=-1)
    return tuple(np.clip(m,0,1).astype(np.float32) for m in (albedo,normal,orm))


def concentration(a):
    # Fraction of luminance spectral energy in the largest 64 non-DC bins.
    # Measures narrow repeated components, not subjective photorealism.
    v=np.asarray(a,float)@np.array([.2126,.7152,.0722]);p=abs(np.fft.fft2(v-v.mean()))**2;p[0,0]=0.;p=p.ravel()
    return float(np.sort(p)[-64:].sum()/p.sum())


def replace_maps(arrays,out):
    root=out/'material_bank';manifest=json.loads((out/'material_bank.json').read_text());checks=[]
    for name in UNSTRUCTURED:
        i=ap.LAYERS[name][0];p=root/name;old=np.asarray(Image.open(p/'basecolor.png').convert('RGB'),np.float32)/255.
        mats=material(name);again=material(name)
        for a,b in zip(mats,again):np.testing.assert_array_equal(a,b);assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
        for j,(k,a) in enumerate(zip(('basecolor','normal','orm'),mats)):
            q=np.rint(a*255).astype(np.uint8);Image.fromarray(q).save(p/(k+'.png'));a=q.astype(np.float32)/255.
            arrays[j][i]=np.dstack((ap.linear(a),np.ones(a.shape[:2],np.float32))) if j==0 else a
        new=np.asarray(Image.open(p/'basecolor.png').convert('RGB'),np.float32)/255.;a,b=concentration(old),concentration(new)
        assert b<a*.65,(name,a,b)
        checks.append(dict(name=name,old_peak_fraction=a,new_peak_fraction=b))
        manifest[i-16]['microtexture']='isotropic seeded FFT random field, stage 28; shared periodic source, decorrelated shader phases'
    (out/'material_bank.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    sheet=Image.new('RGB',(5*240,4*204),(238,239,240));d=ImageDraw.Draw(sheet)
    for i,name in enumerate(ap.BANK):
        im=Image.open(root/name/'basecolor.png').resize((224,176));x=i%5*240;y=i//5*204;sheet.paste(im,(x+8,y));d.text((x+8,y+178),name,fill='black')
    sheet.save(out/'material_contact.png');return checks


def proxies(records,profiles):
    items={};order=sorted(profiles)
    for r in records:
        f=r['footprint'];bid=f['building_id'];off=np.asarray(r['offset']);xy=np.asarray(f['exterior']);h=r['decision']['top_m']
        lo=np.r_[xy.min(0),0.]+off;hi=np.r_[xy.max(0),h]+off;levels=r['architecture']['floor_levels'];fh=float(np.median(np.diff(levels)))
        color=ap.linear(ap.BANK[profiles[bid]['body']][2]);items[bid]=(lo,hi,fh,color)
    out=np.zeros((len(order),12,4),np.float32);manifest=[]
    for i,bid in enumerate(order):
        lo,hi,_,_=items[bid];center=(lo+hi)/2
        near=sorted((k for k in order if k!=bid),key=lambda k:(np.linalg.norm((items[k][0]+items[k][1])[:2]/2-center[:2]),k))[:4]
        for j,k in enumerate(near):
            l,h,fh,col=items[k];out[i,3*j]=[*l,1.];out[i,3*j+1]=[*h,fh];out[i,3*j+2]=[*col,0.]
            assert np.all(h>l) and fh>0
        manifest.append(dict(building=bid,index=i,neighbors=near))
    assert np.isfinite(out).all();return out,manifest,{k:i for i,k in enumerate(order)}


def extend_metadata(old,table,old_table_count,old_remap,profiles,proxy_ids,source_layers):
    # Insert new opaque IDs before the cutout rows. Preserve every old
    # opening, host frame, layer and placement ID; only append proxy index.
    n=len(table)-old_table_count;assert n>0
    m=np.zeros((len(old)+n,14,4),np.float32);m[:old_table_count]=old[:old_table_count];m[len(table):]=old[old_table_count:]
    reference={r['building']:r['id'] for r in table[:old_table_count] if r.get('role')=='wall'}
    for r in table:
        j=r['id'];p=profiles[r['building']]
        if j>=old_table_count:
            m[j,8:]=old[reference[r['building']],8:];fam=r['family'];layer,repeat=source_layers[fam]
            if fam=='metal_roof_red':layer,repeat=ap.LAYERS[p['roof'] if p['roof'].startswith('metal') else 'metal_graphite']
            elif fam=='stone_plinth':layer,repeat=ap.LAYERS['stone_sand']
            elif fam=='concrete_panel':layer,repeat=ap.LAYERS[p['body'] if p['body'].startswith('panel') else 'panel_gray']
            m[j,0]=[0,layer,repeat,ap.stable(r['surface'])%4096];m[j,12,2]={'stage28_entry_cap':10,'stage28_balcony_edge':11,'stage28_sill':12}[r['accessory']]
        m[j,10,3]=proxy_ids[r['building']]
    remap={}
    for r in old_remap:
        bid,kind=r['building'],r['kind'];j=r['id']+n;m[j,10,3]=proxy_ids[bid];remap[bid,kind]=j
    keep=np.ones((14,4),bool);keep[10,3]=False
    np.testing.assert_array_equal(m[:old_table_count][:,keep],old[:old_table_count][:,keep]);np.testing.assert_array_equal(m[len(table):][:,keep],old[old_table_count:][:,keep]);assert len(m)<=16384
    return m,remap
