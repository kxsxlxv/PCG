"""Private local window-box probe. Use a user-licensed EXR directory; never CI."""
import argparse, os, json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from zero_window_materials import load_exr,cross_cells,windowbox,pom,srgb

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    if os.getenv('GITHUB_ACTIONS'):raise RuntimeError('private EXR rendering must not run in CI')
    args.output.mkdir(parents=True,exist_ok=True);fs=sorted(args.source.glob('*.exr*'));assert fs
    W,H=400,320;xx,yy=np.meshgrid((np.arange(W)+.5)/W,1-(np.arange(H)+.5)/H);uv=np.c_[xx.ravel(),yy.ravel()];report=[]
    for path in fs:
        try:a=load_exr(path)
        except ValueError as e:report.append(dict(file=path.name,error=str(e)));continue
        atlas=cross_cells(a);sheet=Image.new('RGB',(1260,350),(226,228,228));draw=ImageDraw.Draw(sheet);images=[]
        w=1.8 if '_S_' in path.name else 3.3 if '_M_' in path.name else 5. if '_L_' in path.name else 2.4
        for j,ratio in enumerate(([0.,0.],[.55,.14],[-.65,.18])):
            q,k,_=pom(uv,ratio,w,2.4,'double',False,False,20);room=windowbox(atlas,q,ratio,3.8,w,2.4,400);assert np.isfinite(room).all()
            c=room*1.6;fr=.045+.55*(1-1/np.sqrt(1+np.dot(ratio,ratio)))**5;c=c*(1-fr)+srgb(np.array([.50,.62,.70]))*fr
            for i,rgb in ((1,[.43,.42,.39]),(2,[.44,.45,.43]),(4,[.62,.64,.63])):c[k==i]=srgb(np.array(rgb))
            c=c/(1+c);rgb=np.uint8(np.clip(c**(1/2.2)*255,0,255)).reshape(H,W,3);images.append(rgb);sheet.paste(Image.fromarray(rgb),(j*420,25));draw.text((j*420+5,6),str(ratio),fill='black')
        sheet.save(args.output/(path.name+'.jpg'),quality=92);report.append(dict(file=path.name,dimensions=list(a.shape),angle_rgb_mae=[float(np.mean(np.abs(images[0].astype(float)-q))) for q in images[1:]]))
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
