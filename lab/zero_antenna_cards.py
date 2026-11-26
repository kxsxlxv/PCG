"""Original antenna impostors: one fixed two-sided quad per instance.
The shared atlas is a replaceable fallback, not a licensed photograph.
"""
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import box
from compile_zero_tiles import opening_rectangles
import zero_facade_trim_geometry as wallgeo
import zero_roof_detail_geometry as roofgeo

N=256
GRID=4
SIZE=N*GRID
# Width and height of the actual card in meters, inclusive of mounting arm.
TYPES=(("yagi",1.58,1.85),("mast",.82,2.1),("loop",1.20,1.72),("double_yagi",1.65,2.12),
       ("dish_white",.91,.86),("dish_grey",.82,.81),("dish_aged",.96,.89),("dish_small",.69,.67))

def atlas():
    out=Image.new("RGBA",(SIZE,SIZE),(116,117,115,0))
    for k,(name,w,h) in enumerate(TYPES):
        scale=3;im=Image.new("RGBA",(N*scale,N*scale),(120,120,117,0));d=ImageDraw.Draw(im)
        def line(points,width,color):
            d.line([(round(x*scale),round(y*scale)) for x,y in points],fill=color,width=max(1,round(width*scale)))
        steel=(126,130,130,255);dark=(66,69,67,255);bright=(190,194,191,255)
        if k<4:
            line([(127,248),(127,17 if k==1 else 40)],4,dark)
            line([(126,248),(126,18 if k==1 else 40)],1.2,bright)
            if k in (0,3):
                line([(32,88),(224,64)],4,steel)
                for j in range(7):
                    x=43+j*25;y=87-j*3
                    l=43-j*3
                    line([(x-8,y-l),(x+8,y+l)],3.3,steel)
                    line([(x-8,y-l),(x+8,y+l)],.85,bright)
                if k==3:
                    line([(35,143),(218,130)],3.5,dark)
                    for j in range(5):
                        x=52+j*37;y=141-j*3
                        line([(x-5,y-23),(x+6,y+23)],3,steel)
            elif k==1:
                line([(57,84),(198,84)],3,steel)
                line([(127,54),(127,15)],2.8,bright)
                for x in (74,179):line([(127,107),(x,84)],2,dark)
            else:
                pts=[(127+53*np.cos(j*2*np.pi/24),83+40*np.sin(j*2*np.pi/24)) for j in range(25)]
                line(pts,4,steel);line([(127,123),(127,44)],3,dark)
            line([(127,201),(109,245)],2.8,dark)
        else:
            # Offset reflector with baked concave gradient, rolled rim and
            # feed arm. Wall bracket reaches the left edge of the card.
            line([(7,162),(43,162),(80,140)],8,dark)
            line([(7,158),(43,158),(80,137)],2.2,steel)
            a=np.asarray(im).copy()
            yy,xx=np.mgrid[0:N*scale,0:N*scale]/scale
            # Oblique elliptical reflector rather than an icon circle.
            ang=(-.24 if k%2==0 else -.10)
            dx=xx-145;dy=yy-121
            X=dx*np.cos(ang)+dy*np.sin(ang);Y=-dx*np.sin(ang)+dy*np.cos(ang)
            rx=(80,67,78,62)[k-4];ry=(99,98,94,83)[k-4]
            rr=np.sqrt((X/rx)**2+(Y/ry)**2)
            inside=rr<=1
            base=np.array([(211,213,208),(138,146,147),(193,187,172),(224,224,219)][k-4],float)
            tint=.88+.16*(Y/ry)+.07*np.cos(np.minimum(rr,1)*np.pi)-.13*np.exp(-((rr-.96)/.045)**2)
            if k==6:tint-=.08*(.5+.5*np.sin(xx*.19+yy*.08))*np.clip(Y/ry+.4,0,1)
            rgb=np.uint8(np.clip(base[None,None,:]*tint[:,:,None],0,255))
            a[inside,:3]=rgb[inside];a[inside,3]=255
            im=Image.fromarray(a);d=ImageDraw.Draw(im)
            line([(142,193),(213,175),(226,146)],7,dark)
            line([(142,190),(213,172),(226,144)],2.8,bright)
            line([(216,144),(238,139)],12,(116,120,119,255))
            line([(219,140),(233,137)],2.4,(186,189,184,255))
        im=im.resize((N,N),Image.Resampling.LANCZOS)
        out.paste(im,((k%GRID)*N,(k//GRID)*N))
    a=np.asarray(out).copy()
    # RGB padding for filtering at alpha edges. Work separately per tile.
    for k in range(len(TYPES)):
        i,j=divmod(k,GRID);q=a[i*N:(i+1)*N,j*N:(j+1)*N]
        valid=q[:,:,3]>=128
        # The color of transparent padding is neutral metal, not black.
        q[~valid,:3]=np.median(q[valid,:3],axis=0).astype(np.uint8)
        assert .005<valid.mean()<.65
    return a

def quad(anchor,direction,k):
    name,w,h=TYPES[k];a=np.asarray(anchor,float);d=np.asarray(direction,float)
    assert abs(np.linalg.norm(d)-1)<1e-6 and abs(d[2])<1e-7
    if k<4:a=a-d*w/2
    v=np.array([a,a+d*w,a+d*w+[0,0,h],a+[0,0,h]],np.float32)
    i,j=divmod(k,GRID)
    u0=(j*N+7.5)/SIZE;u1=(j*N+248.5)/SIZE
    top=(i*N+7.5)/SIZE;bottom=(i*N+248.5)/SIZE
    # Export helper accepts GL convention V up, readback returns image V down.
    uv=np.array([[u0,1-bottom],[u1,1-bottom],[u1,1-top],[u0,1-top]],np.float32)
    f=np.array([[0,1,2],[0,2,3]],np.uint32)
    return v,uv,f

def plan(scene,cells):
    v,uv,ids,table=scene;roof,_,_=roofgeo.parts(scene)
    parts=wallgeo.blockers(scene)
    obstacles=[]
    for b,k,p in parts:
        lo,hi=roofgeo.bbox(p);obstacles.append((b,k,lo,hi))
    accepted=[];attempts=0
    def add(b,name,k,a,d,support):
        nonlocal attempts
        attempts+=1
        q=quad(a,d,k);lo,hi=roofgeo.bbox(q[0])
        for ob,_,l,h in obstacles:
            if ob is not None and ob!=b:continue
            if np.all(np.minimum(hi+.035,h+.035)-np.maximum(lo-.035,l-.035)>0):return False
        for m in accepted:
            l,h=roofgeo.bbox(m["quad"][0])
            if np.all(np.minimum(hi+.07,h)-np.maximum(lo-.07,l)>0):return False
        accepted.append(dict(id=b+":"+name,building=b,type=TYPES[k][0],variant=k,
           support=support,anchor=np.asarray(a).tolist(),quad=q))
        return True
    for b,tris in sorted(roof.items()):
        host=roofgeo.planes(tris);lo,hi=roofgeo.bbox(tris);flat=hi[2]-lo[2]<1e-5
        for i,k in enumerate((0,1,2,3)):
            # Use building-local sequence; alternating roof strips. Fixed
            # azimuth avoids camera-facing rotation and moving shadows.
            found=False
            for fx in (.18+i*.20,.13+i*.23,.24+i*.16):
                for fy in ((.30,.72,.47) if i%2==0 else (.70,.30,.53)):
                    x,y=lo[:2]+np.array([fx,fy])*(hi-lo)[:2]
                    z=roofgeo.height_at(host,x,y)
                    ang=(i*.67+.23) if flat else 0.
                    d=np.array([np.cos(ang),np.sin(ang),0.])
                    # The whole card stays within the host roof footprint.
                    q=quad([x,y,z-.012],d,k)[0]
                    if any(not (lo[0]+.30<=p[0]<=hi[0]-.30 and lo[1]+.30<=p[1]<=hi[1]-.30) for p in q):continue
                    if add(b,"roof_antenna:"+str(i),k,[x,y,z-.012],d,"roof_plane"):
                        found=True;break
                if found:break
    for r in table:
        if r.get("role")!="wall" or "accessory" in r or r.get("party") or not r["bays"]:continue
        frame=wallgeo.wall_frame(v,uv,ids,r);o,t,n=frame
        ops=opening_rectangles(r,cells)
        if not ops:continue
        wanted=2 if r["length"]>=20 else 1
        made=0
        # Place against solid piers between real openings, high enough for
        # upper-floor dishes but not at the parapet or in a window aperture.
        for f in range(r["floors"]-1,0,-1):
            if made>=wanted:break
            for bay in range(r["bays"]-1):
                k=4+(r["seed"]+f+bay)%4;_,w,h=TYPES[k]
                gap=r["first_center"]+(bay+.5)*r["spacing"]
                angle=(.48,.70,.90)[(r["seed"]+bay)%3]
                d=t*np.cos(angle)+n*np.sin(angle)
                u=gap-w*np.cos(angle)/2
                z=f*r["floor_h"]+r["sill"]+.23
                if u<.12 or u+w*np.cos(angle)>r["length"]-.12 or z+h>r["height"]-.22:continue
                if any(min(u+w*np.cos(angle),c+.045)>max(u,a-.045) and min(z+h,e+.045)>max(z,b-.045)
                       for _,(a,b,c,e) in ops.values()):continue
                a=o+u*t+.006*n+np.array([0,0,z])
                if add(r["building"],r["facade_id"]+":dish:"+str(f)+":"+str(bay),k,a,d,"facade_pier"):
                    made+=1
                    if made>=wanted:break
    assert len({m["id"] for m in accepted})==len(accepted)
    assert all(len(m["quad"][2])==2 for m in accepted)
    return accepted,attempts

def batch(cards):
    v=[];u=[];f=[];offset=0
    for m in cards:
        a,b,c=m["quad"];v.append(a);u.append(b);f.append(c+offset);offset+=len(a)
    return np.concatenate(v),np.concatenate(u),np.concatenate(f)
