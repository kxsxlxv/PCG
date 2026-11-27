import json,struct,numpy as np
from pathlib import Path
from collections import Counter
P=Path('artifacts/zero-window-entrance-stage')
def glb(path):
 b=path.read_bytes();n=struct.unpack_from('<I',b,12)[0];t=json.loads(b[20:20+n]);blob=b[28+n:]
 def read(i):
  a=t['accessors'][i];v=t['bufferViews'][a['bufferView']];return np.frombuffer(blob,dtype={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']],count=a['count']*{'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']],offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'],-1)
 return [(p,{k:read(i) for k,i in p['attributes'].items()},read(p['indices']).ravel(),t) for m in t['meshes'] for p in m['primitives']]
def z_up(p):return p[:,[0,2,1]]*np.array([1,-1,1])
def load_scene():
 table=json.loads((P/'parameters.json').read_text());v=[];uv=[];ids=[]
 for p,a,f,t in glb(P/'tile_opaque.glb'):
  v.extend(z_up(a['POSITION'])[f]);uv.extend(a['TEXCOORD_1'][f]);c=a['COLOR_0'].astype(int);ids.extend((c[:,0]+256*c[:,1])[f])
 for path in ('roof_balcony_cutout.glb','antenna_cutout.glb'):
  for p,a,f,t in glb(P/path):
   v.extend(z_up(a['POSITION'])[f]);uv.extend(a['TEXCOORD_0'][f]);ids.extend([len(table)]*len(f));table.append(dict(id=len(table),accessory=t['materials'][p['material']]['name']))
 return (*map(np.array,(v,uv,ids)),table),np.frombuffer((P/'facade_cells.bin').read_bytes(),dtype=np.uint8)
