"""Private roughness-based candidate segmentation; never an automatic UV binding."""
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def candidates(gray,threshold=24,block=8):
    a=np.asarray(gray,dtype=np.float32)
    if a.ndim!=2 or not a.size:raise ValueError('grayscale image required')
    h,w=a.shape
    if h%block or w%block:raise ValueError('image dimensions not divisible by block')
    g=a.reshape(h//block,block,w//block,block).mean(axis=(1,3))
    mask=g<threshold;seen=np.zeros(mask.shape,bool);H,W=mask.shape;rows=[]
    for y,x in np.argwhere(mask):
        if seen[y,x]:continue
        queue=deque([(int(y),int(x))]);seen[y,x]=True
        loX=hiX=int(x);loY=hiY=int(y);n=0
        while queue:
            yy,xx=queue.popleft();n+=1
            loX=min(loX,xx);hiX=max(hiX,xx);loY=min(loY,yy);hiY=max(hiY,yy)
            for a0,b0 in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0<=a0<H and 0<=b0<W and mask[a0,b0] and not seen[a0,b0]:
                    seen[a0,b0]=True;queue.append((a0,b0))
        dx,dy=hiX-loX+1,hiY-loY+1;fill=n/(dx*dy)
        rows.append(dict(rect_px=[loX*block,loY*block,(hiX+1)*block,(hiY+1)*block],blocks=n,fill=fill,candidate=dx>=4 and dy>=4 and n>=20 and fill>=.75))
    return dict(width=w,height=h,block=block,threshold=threshold,dark_fraction=float(mask.mean()),components=len(rows),rectangular_candidates=sum(x['candidate'] for x in rows),regions=rows)

def analyze(path,out):
    with Image.open(path) as im:gray=np.asarray(im.convert('L'))
    result=candidates(gray)
    # Private diagnostic preview only: top-left pixel coordinates. It does not
    # infer material IDs, distinguish glass from other smooth surfaces, or bind maps.
    im=Image.fromarray(gray).convert('RGB');d=ImageDraw.Draw(im)
    for r in result['regions']:
        if r['candidate']:d.rectangle(r['rect_px'],outline=(0,190,255),width=4)
    im.thumbnail((1280,1280));im.save(Path(out)/'roughness_candidates.png')
    result['limitation']='Dark rectangular roughness regions are candidate panes, not verified unique glass/window variants. Pixel block means may differ from JPEG DC means.'
    return result
