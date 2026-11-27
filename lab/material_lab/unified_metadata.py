"""Generalized GPU metadata on PCGF v1 cells and explicit paired sidecar.
The legacy GPU renderer's three hard-coded fixture styles stay unchanged.
"""
import hashlib
import numpy as np
from loggia import fit, affine

LAYERS={'red_brick':(0,1.2),'concrete_panel':(2,2.4),'plaster_cream':(3,2.),
        'stone_plinth':(4,1.6),'bitumen_roof':(5,2.),'ceramic_roof':(14,1.2),
        'painted_trim':(7,1.),'painted_rail':(8,1.),'metal_roof_red':(13,2.4),
        'plaster_white':(10,2.),'wood_door':(15,1.)}
STYLE={'grid':3,'blind_end':4,'sections':6,'ribbon':7,'corner':3,'paired':9}
PROFILES={'baltic_timber':11,'nordic_cottage':3,'central_townhouse':5,
          'mediterranean_villa':8,'caucasus_house':4,'panel_slab':8,
          'brick_walkup':3,'perimeter_block':5,'mediterranean_balcony':8,
          'caucasus_tower':0,'modern_gallery':2,'masonry_office':5,
          'modern_ribbon':12,'glass_grid':14,'civic_block':9}


def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little')%4096


def build(scene,cells,pairs,styles):
    v,u,ids,table=scene;models=fit(scene);by={d['record']['id']:m for m in models.values() for d in m['faces'].values()}
    meta=np.zeros((len(table),8,4),np.float32);rows=[];checked=0;paired=0;states=[0,0,0]
    for r in table:
        j=r['id'];assert 0<=j<len(table);k=r.get('accessory','');mask=ids==j
        if k.startswith('cutout_'):
            meta[j,0,:2]=[3,('cutout_roof','cutout_balcony','cutout_antenna').index(k)];continue
        layer,repeat=LAYERS[r['family']];meta[j,0]=[0,layer,repeat,seed(r.get('surface',k))]
        if not mask.any():continue
        if k in ('coping','gutter_side','gutter_floor','bent_drain','flashing','flue','roof24_drain'):meta[j,0,1:3]=[9,1.]
        if k=='vent':meta[j,0,1:3]=[12,1.]
        if k=='ac':
            w,h=r['span'];meta[j,0,0]=5;meta[j,1,:3]=[w,h,float(w>.45 and h>.40)];continue
        if k in ('dormer_front','access_front'):
            meta[j,0,0]=6 if k=='dormer_front' else 7;meta[j,1,:2]=r['span'];continue
        if k=='roof24_drain':
            xyz=v[mask][:3];n=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);n/=np.linalg.norm(n)
            if n[2]>.9:meta[j,0,0]=8
            continue
        if j in by:
            m=by[j];host=m['host'];meta[j,0,0]=2;key=seed(r['placement'])
            meta[j,1,:3]=[host['id'],(.27,.34,.40)[key%3],(key%100)/100.]
            meta[j,2,:3]=m['origin'];meta[j,3,:3]=m['frame'][:,0];meta[j,4,:3]=m['frame'][:,1]
            meta[j,5,:3]=m['lo'];meta[j,6,:3]=m['hi'];meta[j,7,:2]=r['span'];continue
        if r.get('role')!='wall':continue
        a=styles[r['building']];name=a['id'];style=STYLE[a['composition']]
        if name in ('baltic_timber','nordic_cottage'):style=5
        elif name=='glass_grid':style=8
        elif name in ('brick_walkup','masonry_office','civic_block','central_townhouse','perimeter_block'):style=9 if style!=6 else 6
        bays,floors=r['bays'],r['floors'];start=len(rows)
        meta[j,0,:3]=[1,layer,repeat];meta[j,1]=[start,bays,floors,style];meta[j,2,:3]=[r['first_center'],r['spacing'],r['floor_h']]
        meta[j,3,:3]=a['accent'];meta[j,6]=[r['height'],r['window_width'],r['sill'],r['window_height']];meta[j,7,0]=PROFILES[name]
        A=affine(v[mask],u[mask]);t=A[0]/np.linalg.norm(A[0]);n=np.cross(t,A[1]);n/=np.linalg.norm(n)
        np.testing.assert_allclose(A[1],[0.,0.,1.],atol=5e-5,rtol=0);meta[j,4,:3]=t;meta[j,5,:3]=n
        for slot in range(bays*floors):
            q=np.zeros((3,4),np.float32);code=int(cells[r['cell_start']+slot]);role=code&3;state=code>>2
            f,b=divmod(slot,bays);key=seed(r['facade_id']+':'+str(slot));pair=pairs.get(str(j),{}).get(str(slot));center=r['first_center']+b*r['spacing'];w=r['window_width']
            if pair:
                assert role==2 and pair['code']==code
                for rect in (pair['door'],pair['window']):
                    assert center-r['spacing']/2<rect[0]<rect[2]<center+r['spacing']/2
                    assert f*r['floor_h']<rect[1]<rect[3]<(f+1)*r['floor_h']
                q[0]=pair['door'];q[1]=pair['window'];q[2]=[2,1,key,state+1];paired+=1
            elif role:
                assert role in (1,2)
                z=f*r['floor_h']+(r['balcony_threshold'] if role==2 else r['sill']);h=r['door_height'] if role==2 else r['window_height']
                q[0]=[center-w/2,z,center+w/2,z+h];q[2]=[role,0,key,state+1]
            if role:
                assert state<3;states[state]+=1
            rows.append(q);checked+=1
        q=np.zeros((3,4),np.float32)
        if r['door'] is not None:
            x,y,w,h=r['door'];q[0]=[x,y,x+w,y+h];role=6 if a['use']=='house' else 3 if a['use']=='office' or name=='modern_gallery' else 4
            q[2]=[role,0,seed(r['facade_id']+':entry'),4 if role==3 else 1]
        rows.append(q)
    ops=np.asarray(rows,np.float32);assert np.isfinite(meta).all() and np.isfinite(ops).all()
    assert checked==len(cells) and paired==sum(len(x) for x in pairs.values())
    return meta,ops,models,dict(cells=checked,paired=paired,night_states=states,metadata_rows=len(meta),opening_rows=len(ops))
