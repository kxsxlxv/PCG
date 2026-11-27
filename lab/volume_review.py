"""Emit bounded review crops from the actual output PNGs, not mock images."""
import base64, hashlib, io, json, sys
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1]);manifest=json.loads((root/'bundle.json').read_text())
for i in range(2):
    name=f'comparison_{i}.png';p=root/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name]
    im=Image.open(p).convert('RGB')
    for j in range(2):
        crop=im.crop((0,j*1144,im.width,min((j+1)*1144,im.height)))
        crop.thumbnail((1100,1100))
        b=io.BytesIO();crop.save(b,format='JPEG',quality=78,optimize=True)
        s=base64.b64encode(b.getvalue()).decode('ascii');key=f'{i}_{j}'
        print('VOLUME_IMAGE_BEGIN',key,len(s),flush=True)
        for k in range(0,len(s),2048):print('VOLUME_IMAGE_CHUNK',key,s[k:k+2048],flush=True)
        print('VOLUME_IMAGE_END',key,flush=True)
