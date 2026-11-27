"""Measured envelope grammar and provisional atlas-to-world correspondence.
Source meshes are observations, never copied into generated geometry.
The atlas is explicitly provisional when glTF lacks material bindings.
"""
import argparse,json,math,os,tempfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
from PIL import Image,ImageDraw
from shapely import polygons,points,constrained_delaunay_triangles
from shapely.geometry import box
from shapely.strtree import STRtree
from reference_intake33 import Gltf,obj_groups,extract,sha,dump,measure
from reference_atlas33 import candidates
from reference_contours33 import fit_contours,bounded_mesh

def parts_arrays(parts):
    T=[];U=[]
    for p in parts:
        t=p.xyz[p.faces];u=p.uv[p.faces] if p.uv is not None else np.full((len(t),3,2),np.nan)
        T.append(t);U.append(u)
    t=np.concatenate(T);uv=np.concatenate(U)
    span=np.ptp(t.reshape(-1,3),axis=0);n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    a=np.linalg.norm(n,axis=1);ok=a>2*max(float(span.max()),1e-12)**2*1e-12
    return t[ok],uv[ok],n[ok]/a[ok,None],a[ok]*.5

def frame(t,n,a):
    # Axes are estimated from area-weighted vertical normals modulo 90 degrees.
    m=abs(n[:,2])<.10
    ang=np.arctan2(n[m,1],n[m,0]);v=(a[m]*np.exp(4j*ang)).sum()
    theta=float(np.angle(v)/4) if abs(v)>1e-12 else 0.
    c,s=math.cos(theta),math.sin(theta);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    q=t@R;lo=q.reshape(-1,3).min(0);hi=q.reshape(-1,3).max(0)
    return (q-lo),n@R,hi-lo,theta,float(abs(v)/max(a[m].sum(),1e-12))

def qweight(x,w,p):
    i=np.argsort(x,kind='stable');x=x[i];w=w[i];c=np.cumsum(w)
    return float(np.interp(p*c[-1],c,x))

def slice_box(t,z):
    m=(t[:,:,2].min(1)<z)&(t[:,:,2].max(1)>z);v=t[m]
    if not len(v):return None
    # Each nonhorizontal triangle crosses the slice at two edges.
    pts=[];ids=[]
    for i,j in ((0,1),(1,2),(2,0)):
        a,b=v[:,i],v[:,j];hit=(a[:,2]-z)*(b[:,2]-z)<0
        ii=np.flatnonzero(hit);a,b=a[hit],b[hit]
        q=a[:,:2]+(b[:,:2]-a[:,:2])*((z-a[:,2])/(b[:,2]-a[:,2]))[:,None]
        pts.append(q);ids.append(ii)
    p=np.concatenate(pts);idx=np.concatenate(ids)
    if len(p)<4:return None
    order=np.argsort(idx,kind='stable');p=p[order];idx=idx[order]
    if len(p)%2 or not np.array_equal(idx[::2],idx[1::2]):return None
    w=np.repeat(np.linalg.norm(p[::2]-p[1::2],axis=1),2)
    if w.sum()<=1e-12:return None
    return np.array([qweight(p[:,0],w,.015),qweight(p[:,1],w,.015),qweight(p[:,0],w,.985),qweight(p[:,1],w,.985)])

def fit_tiers(t,span,count=32):
    if min(span)<=1e-9:return []
    scale=np.array([span[0],span[1],span[0],span[1]])
    samples=[]
    for i in range(count):
        r=slice_box(t,(i+.473)*span[2]/count)
        if r is not None:
            r=np.clip(r/scale,0,1)
            if min(r[2:]-r[:2])>=.06:samples.append((i,r))
    if len(samples)<count*.65:return []
    groups=[]
    for i,r in samples:
        if groups and np.max(abs(r-np.median([x[1] for x in groups[-1]],axis=0)))<=.035:
            groups[-1].append((i,r))
        else:groups.append([(i,r)])
    # Merge single-slice excursions into nearest neighbour; do not model spikes.
    while len(groups)>1 and (min(map(len,groups))<2 or len(groups)>8):
        i=min(range(len(groups)),key=lambda j:(len(groups[j]),j))
        neighbors=[j for j in (i-1,i+1) if 0<=j<len(groups)]
        j=min(neighbors,key=lambda j:float(np.linalg.norm(np.median([x[1] for x in groups[i]],0)-np.median([x[1] for x in groups[j]],0))))
        a,b=sorted((i,j));groups[a:b+1]=[groups[a]+groups[b]]
    tiers=[];start=0.
    for i,g in enumerate(groups):
        end=1. if i==len(groups)-1 else (g[-1][0]+groups[i+1][0][0]+1)/2/count
        r=np.median([x[1] for x in g],axis=0)
        if end>start:tiers.append(dict(z0=start,z1=end,rect=r.tolist(),samples=len(g)))
        start=end
    return tiers

def mesh_tiers(tiers,span):
    """Reusable normalized cross-section grammar, no source triangles."""
    out=[];uv=[];role=[]
    def add(t,u,k):out.append(np.asarray(t,float));uv.append(np.asarray(u,float));role.append(k)
    def cap(p,z,up):
        if p.is_empty:return
        for g in constrained_delaunay_triangles(p).geoms:
            xy=np.asarray(g.exterior.coords)[:3];t=np.c_[xy,np.full(3,z)]
            if (np.cross(t[1]-t[0],t[2]-t[0])[2]>0)!=up:t=t[::-1]
            add(t,t[:,:2]/span[:2],0)
    boxes=[]
    for q in tiers:
        r=np.array(q['rect'])*np.tile(span[:2],2);boxes.append(box(*r))
    for i,(q,p) in enumerate(zip(tiers,boxes)):
        z,Z=q['z0']*span[2],q['z1']*span[2]
        ring=np.asarray(p.exterior.coords)
        if not p.exterior.is_ccw:ring=ring[::-1]
        for a,b in zip(ring[:-1],ring[1:]):
            w=np.linalg.norm(b-a);v=np.array([[*a,z],[*b,z],[*b,Z],[*a,Z]])
            u=np.array([[0,z],[w,z],[w,Z],[0,Z]])
            for f in ([0,1,2],[0,2,3]):add(v[f],u[f],1)
        cap(p if i==0 else p.difference(boxes[i-1]),z,False)
        cap(p if i==len(boxes)-1 else p.difference(boxes[i+1]),Z,True)
    return np.array(out),np.array(uv),np.array(role)

def atlas_centers(t,uv,n,rects,size):
    ok=(abs(n[:,2])<.10)&np.isfinite(uv).all(axis=(1,2))
    ok &= ((uv>=0)&(uv<=1)).all(axis=(1,2))
    a=uv[:,1]-uv[:,0];b=uv[:,2]-uv[:,0];det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    ok &= abs(det)>1e-12
    tt=t[ok];uu=uv[ok];nn=n[ok]
    if not len(tt):return []
    # Spatial query on UV triangles, then barycentric inversion in real geometry.
    poly=polygons(uu);tree=STRtree(poly)
    rects=np.asarray(rects,float)/np.tile(size,2)
    center=(rects[:,:2]+rects[:,2:])/2
    pairs=tree.query(points(center),predicate='intersects')
    rows=[];seen=set()
    for r,i in zip(*pairs):
        A=np.stack((uu[i,1]-uu[i,0],uu[i,2]-uu[i,0]),axis=1)
        W=np.stack((tt[i,1]-tt[i,0],tt[i,2]-tt[i,0]),axis=1)
        J=W@np.linalg.inv(A);p=tt[i,0]+J@(center[r]-uu[i,0])
        dims=rects[r,2:]-rects[r,:2]
        width=float(np.linalg.norm(J[:2,0])*dims[0]+np.linalg.norm(J[:2,1])*dims[1]);height=float(abs(J[2])@dims)
        axis=int(np.argmax(abs(nn[i,:2])));side=1 if nn[i,axis]>0 else -1
        key=(int(r),axis,side,*np.round(p,5))
        if key in seen:continue
        seen.add(key);rows.append(dict(atlas_region=int(r),center=p.tolist(),axis=axis,side=side,width_estimate=float(width),height_estimate=float(height)))
    return rows

def lattice(rows,span):
    # Reject implausibly tiny/large affine extrapolations. Periods are hypotheses.
    h=span[2];q=[r for r in rows if .01*h<r['height_estimate']<.45*h and .005*h<r['width_estimate']<.6*max(span[:2])]
    def period(values,tol):
        if len(values)<3:return None
        a=np.sort(values);bins=[float(a[0])]
        for x in a[1:]:
            if x-bins[-1]>tol:bins.append(float(x))
        d=np.diff(bins);d=d[d>tol]
        if len(d)<2:return None
        # Prefer repeated adjacent separation, not the smallest gap.
        step=max(tol,1e-9);k=np.round(d/step).astype(int);v,c=np.unique(k,return_counts=True)
        return float(np.median(d[k==v[np.argmax(c)]])) if c.max()>=2 else None
    z=period([r['center'][2] for r in q],h*.018)
    if z and q and z<1.10*np.median([r['height_estimate'] for r in q]):z=None
    xs=[]
    for axis in (0,1):
        for side in (-1,1):
            p=period([r['center'][1-axis] for r in q if r['axis']==axis and r['side']==side],h*.018)
            if p and q and p>=1.10*np.median([r['width_estimate'] for r in q]):xs.append(p)
    return dict(valid_affine_hypotheses=len(q),vertical_period=z,horizontal_period=float(np.median(xs)) if xs else None,
      window_width=float(np.median([r['width_estimate'] for r in q])) if q else None,
      window_height=float(np.median([r['height_estimate'] for r in q])) if q else None)

class Renderer:
    def __init__(self,atlas):
        import moderngl,ctypes.util
        self.ctx=moderngl.create_context(standalone=True,backend='egl',require=330,libegl=ctypes.util.find_library('EGL'),libgl=ctypes.util.find_library('GL'))
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.tex=self.ctx.texture(atlas.size,3,atlas.tobytes());self.tex.use()
        self.program=self.ctx.program(vertex_shader="""#version 330
in vec3 p;in vec3 n;in vec2 uv;in float role;
uniform vec3 center,right,up,view;uniform vec2 extent;uniform float depth;
out vec3 N;out vec2 U;out float R;
void main(){vec3 q=p-center;gl_Position=vec4(dot(q,right)/extent.x,dot(q,up)/extent.y,-dot(q,view)/depth,1);N=n;U=uv;R=role;}
""",fragment_shader="""#version 330
in vec3 N;in vec2 U;in float R;out vec4 color;
uniform sampler2D atlas;uniform int mode;uniform vec4 grid;
void main(){
vec3 n=normalize(N);vec3 c=vec3(.69,.67,.62);
if(mode==1)c=texture(atlas,clamp(U,0.,1.)).rgb;
if(mode==2&&R>.5&&grid.x>0.&&grid.y>0.){
vec2 q=abs(fract(U/ grid.xy)-.5)*grid.xy;
if(q.x<grid.z*.5&&q.y<grid.w*.5)c=vec3(.19,.28,.32);
}
float l=.43+.57*abs(dot(n,normalize(vec3(.4,-.5,.75))));
color=vec4(c*l,1.);
}""")
    def draw(self,t,uv,role,center,extent,view,mode,grid,res):
        n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)
        data=np.c_[t.reshape(-1,3),np.repeat(n,3,axis=0),np.nan_to_num(uv).reshape(-1,2),np.repeat(role,3)].astype('f4')
        b=self.ctx.buffer(data.tobytes());vao=self.ctx.vertex_array(self.program,[(b,'3f 3f 2f 1f','p','n','uv','role')])
        f=self.ctx.simple_framebuffer(res,components=4);f.use();f.clear(.91,.93,.95,0.,depth=1.)
        view=np.array(view,float);view/=np.linalg.norm(view);right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
        for k,v in dict(center=center,right=right,up=up,view=view,extent=extent,depth=float(max(extent)*4),mode=mode,grid=grid).items():self.program[k].value=tuple(v) if np.ndim(v) else v
        vao.render();im=Image.frombytes('RGBA',res,f.read(components=4)).transpose(Image.Transpose.FLIP_TOP_BOTTOM);vao.release();b.release();f.release()
        return im

def run(low,high,albedo,rough,out):
    out.mkdir(parents=True,exist_ok=False)
    atlas=Image.open(albedo).convert('RGB');gray=Image.open(rough).convert('L')
    assert atlas.size==gray.size
    rects=[x['rect_px'] for x in candidates(np.asarray(gray))['regions'] if x['candidate']]
    renderer=Renderer(atlas);rows=[];selected=[]
    def audit(key,groups,highpack):
        for i,(name,parts) in enumerate(groups):
            t,uv,n,a=parts_arrays(parts);t,n,span,theta,confidence=frame(t,n,a)
            k=f'{key}_{i:03d}';tiers=fit_tiers(t,span)
            contours,closed=fit_contours(t,n,a,span,tiers,repair_open=highpack) if tiers else ([],0)
            matches=atlas_centers(t,uv,n,rects,atlas.size) if highpack else []
            lat=lattice(matches,span)
            row=dict(id=k,name=name,pack='high' if highpack else 'low',source_triangles=len(t),span_source_units=span.tolist(),orientation_rad=theta,orthogonal_confidence=confidence,
                     vertical_area_fraction=float(a[abs(n[:,2])<.1].sum()/a.sum()),tiers=tiers,contours=contours,closed_slices=closed,atlas_matches=len(matches),lattice=lat)
            rows.append(row)
            dump(out/(k+'_modules.json'),dict(provisional_atlas=highpack,matched_centers=matches,model=row))
            if not tiers or min(span)<=0 or row['vertical_area_fraction']<.22 or min(span[:2])/span[2]<.12:return_flag=False
            else:return_flag=True
            if return_flag:
                # A deterministic sample of all eligible groups, not a best-fit ranking.
                selected.append((row,t,uv,span,highpack))
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);extract(low,root)
        for i,p in enumerate(sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.gltf','.glb','.obj'))):
            audit(f'low_{i:03d}',Gltf(p).groups() if p.suffix.lower()!='.obj' else obj_groups(p),False)
        audit('high',Gltf(high).groups(),True)
    report=[]
    for pack in ('low','high'):
        eligible=[x for x in selected if x[0]['pack']==pack]
        if not eligible:continue
        # sample six across height/aspect distribution, deterministically
        eligible.sort(key=lambda x:(x[3][2]/max(x[3][:2]),x[0]['id']))
        indices=np.unique(np.round(np.linspace(0,len(eligible)-1,min(6,len(eligible)))).astype(int))
        for idx in indices:
            row,t,uv,span,hp=eligible[idx];k=row['id'];base,bu,br=mesh_tiers(row['tiers'],span)
            fitted=bounded_mesh(row['contours'],span) if row['contours'] else None
            g,gu,gr,tol=fitted if fitted is not None else (base,bu,br,None)
            assert len(g)<=1800 and len(g)>0
            import trimesh
            m=trimesh.Trimesh(vertices=g.reshape(-1,3),faces=np.arange(g.size//3).reshape(-1,3),process=True)
            m.export(out/(k+'_grammar.glb'))
            read=trimesh.load(out/(k+'_grammar.glb'),force='mesh',process=False)
            assert len(read.faces)==len(g)
            # Same camera and bounds for source and grammar; neutral silhouette test.
            res=(1280,960);center=span/2;extent=(float(np.linalg.norm(span)*.60*res[0]/res[1]),float(np.linalg.norm(span)*.60))
            scores=[];baseline_scores=[];sheet=Image.new('RGB',(res[0]*3,(res[1]+35)*2),(235,237,240));d=ImageDraw.Draw(sheet)
            for j,view in enumerate(((1,-1.7,.65),(-1,1.7,.65))):
                a=renderer.draw(t,uv,np.zeros(len(t)),center,extent,view,0,(0,0,0,0),res)
                b=renderer.draw(g,gu,gr,center,extent,view,0,(0,0,0,0),res)
                A=np.asarray(a)[:,:,3]>0;B=np.asarray(b)[:,:,3]>0
                bb=renderer.draw(base,bu,br,center,extent,view,0,(0,0,0,0),res);BB=np.asarray(bb)[:,:,3]>0
                baseline_scores.append(float((A&BB).sum()/max(1,(A|BB).sum())))
                scores.append(float((A&B).sum()/max(1,(A|B).sum())))
                for l,im in enumerate((a,bb,b)):
                    sheet.paste(im.convert('RGB'),(l*res[0],j*(res[1]+35)));d.text((l*res[0]+8,j*(res[1]+35)+res[1]+7),('Source geometry' if l==0 else 'Rectangular tiers' if l==1 else 'Measured contour grammar')+f' | view {j+1}',fill='black')
            sheet.save(out/(k+'_comparison.png'))
            if hp:
                # Explicitly provisional external atlas, not certified original PBR.
                a=renderer.draw(t,uv,np.zeros(len(t)),center,extent,(1,-1.7,.65),1,(0,0,0,0),res)
                a.save(out/(k+'_provisional_albedo.png'))
                lat=row['lattice'];grid=(lat['horizontal_period'] or 0,lat['vertical_period'] or 0,lat['window_width'] or 0,lat['window_height'] or 0)
                renderer.draw(g,gu,gr,center,extent,(1,-1.7,.65),2,grid,res).save(out/(k+'_lattice.png'))
            report.append(dict(id=k,pack=pack,source_triangles=len(t),grammar_triangles=len(g),baseline_triangles=len(base),silhouette_iou=scores,baseline_iou=baseline_scores,tiers=len(row['contours']),closed_slices=row['closed_slices'],contour_tolerance=tol,lattice=row['lattice']))
    result=dict(schema='pcg.reference.measured-envelope.v1',groups=len(rows),eligible=len(selected),atlas_regions=len(rects),
      matches=sum(x['atlas_matches'] for x in rows),models=rows,comparisons=report,
      limitations=['Polygonized cross-section proxies with explicit rectangular fallback, not authoritative GIS outlines or finished PCG buildings.','Only neutral geometry IoU is compared; pitched roofs, open walls and small roof details may be poorly approximated.','External high atlas is provisionally sampled, no certified material binding.','Lattice dimensions are affine hypotheses; no full PBR baking or complete module grammar.','Source units not independently calibrated.'])
    dump(out/'report.json',result)
    print('MATCH33',json.dumps(dict(groups=len(rows),eligible=len(selected),matches=result['matches'],comparisons=report)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('low','high','albedo','rough','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.low,a.high,a.albedo,a.rough,a.out)
