"""Research-only geometry probe. XY metres, Z up; no UNIGINE dependency.
Requires shapely>=2.1, numpy, trimesh. Does NOT implement straight skeleton.
"""
import argparse, json, math, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import shapely as s
from shapely.geometry import Polygon, LineString, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import split, triangulate
import trimesh


def polys(g):
    if g.geom_type == 'Polygon':
        if not g.is_empty and g.area > 1e-10: yield g
    elif hasattr(g,'geoms'):
        for x in g.geoms: yield from polys(x)


def cdt(p): return list(polys(s.constrained_delaunay_triangles(p)))


def examples():
    return {
        'rectangle':box(0,0,12,8),
        'L':Polygon([(0,0),(12,0),(12,5),(6,5),(6,10),(0,10)]),
        'U':Polygon([(0,0),(12,0),(12,10),(8,10),(8,4),(4,4),(4,10),(0,10)]),
        'T':Polygon([(0,0),(12,0),(12,4),(8,4),(8,10),(4,10),(4,4),(0,4)]),
        'courtyard':Polygon([(0,0),(16,0),(16,12),(0,12)], [[(4,3),(12,3),(12,9),(4,9)]]),
        'narrow':box(0,0,20,.5),
        'irregular':Polygon([(0,0),(7,-1),(12,3),(10,7),(13,10),(3,9)]),
        'notch':Polygon([(0,0),(12,0),(12,12),(7,12),(7,2),(6.9,2),(6.9,12),(0,12)]),
        'collinear':Polygon([(0,0),(3,0),(6,0),(12,0),(12,8),(0,8)]),
        'bowtie':Polygon([(0,0),(8,8),(0,8),(8,0)])}


def tri_metrics(p, tris):
    u=s.union_all(tris)
    return {'triangles':len(tris),'missing_m2':round(p.difference(u).area,9),
            'outside_m2':round(u.difference(p).area,9),
            'overlap_m2':round(sum(t.area for t in tris)-u.area,9)}


def cut_line(a,b,c,p):
    # line ax+by+c=0, long enough for bounding box
    v=np.array([a,b]); norm=np.linalg.norm(v)
    if norm<1e-12:return None
    v=v/norm; c=c/norm
    center=np.array(p.centroid.coords[0]); on=center-(center@v+c)*v
    d=np.array([-v[1],v[0]])
    span=4*math.hypot(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1])+1
    return LineString([on-span*d,on+span*d])


def roof_planes(p, mode, pitch=30):
    k=math.tan(math.radians(pitch))
    if mode=='flat':return [np.array([0.,0.,0.])]
    if mode=='gable':
        # fixed X ridge for research; not automatic architecture inference
        lo,hi=p.bounds[1],p.bounds[3]
        return [np.array([0,k,-k*lo]),np.array([0,-k,k*hi])]
    if mode=='hip_convex':
        if p.interiors or p.convex_hull.symmetric_difference(p).area>1e-9:
            raise ValueError('convex hip only')
        ring=list(orient(p,1).exterior.coords); out=[]
        for a,b in zip(ring,ring[1:]):
            d=np.array(b)-a; n=np.array([-d[1],d[0]])/np.linalg.norm(d)
            out.append(np.array([k*n[0],k*n[1],-k*(n@a)]))
        return out
    raise ValueError(mode)


def mesh_build(p,mode,wall=6.):
    p=orient(p,1); planes=roof_planes(p,mode)
    cells=[p]
    for i,pa in enumerate(planes):
        for pb in planes[:i]:
            line=cut_line(*(pa-pb),p)
            if line is not None:cells=[x for cell in cells for x in polys(split(cell,line))]
    def z(x,y):return wall+min(a*x+b*y+c for a,b,c in planes)
    verts=[]; faces=[]; roles=[]; ids={}
    def vid(v):
        key=tuple(round(float(x),8) for x in v)
        if key not in ids:ids[key]=len(verts);verts.append(key)
        return ids[key]
    def face(v,role):faces.append(tuple(vid(x) for x in v));roles.append(role)
    roof_tris=[]
    for cell in cells:
        for t in cdt(cell):
            coords=list(orient(t,1).exterior.coords)[:-1]; roof_tris.append(t)
            face([(x,y,z(x,y)) for x,y in coords],'roof')
    edge_count=Counter(); directed={}
    for t in roof_tris:
        c=[tuple(round(float(v),8) for v in xy) for xy in orient(t,1).exterior.coords]
        for a,b in zip(c,c[1:]):
            key=tuple(sorted([a,b]));edge_count[key]+=1;directed[key]=(a,b)
    border=[directed[e] for e,n in edge_count.items() if n==1]
    # Roof CCW boundary: interior is left, outward wall normal is right.
    for a,b in border:
        ax,ay=a;bx,by=b; a0=(ax,ay,0);b0=(bx,by,0);a1=(ax,ay,z(ax,ay));b1=(bx,by,z(bx,by))
        face([a0,b0,b1],'wall');face([a0,b1,a1],'wall')
    # Node the base to exactly the same boundary positions as roof/wall.
    # Polygonize returns courtyard interiors too; retain only occupied regions.
    lines=[LineString([a,b]) for a,b in border]
    bases=[q for q in polys(s.polygonize(lines)) if p.covers(q.representative_point())]
    for q in bases:
        for t in cdt(q):
            face([(x,y,0) for x,y in list(orient(t,-1).exterior.coords)[:-1]],'base')
    mesh=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    colors={'roof':[153,81,58,255],'wall':[219,205,177,255],'base':[110,110,110,255]}
    mesh.visual.face_colors=np.array([colors[r] for r in roles],dtype=np.uint8)
    m=tri_metrics(p,roof_tris)
    # Exact for each affine triangle: integral of z = area * mean(vertex z).
    expected=sum(t.area*sum(z(x,y) for x,y in list(t.exterior.coords)[:-1])/3 for t in roof_tris)
    m.update(mesh_triangles=len(faces),vertices=len(verts),watertight=bool(mesh.is_watertight),
             winding_consistent=bool(mesh.is_winding_consistent),positive_volume=bool(mesh.volume>0),
             volume_m3=round(float(mesh.volume),8),volume_error_m3=round(abs(float(mesh.volume)-expected),8),
             roof_rise_m=round(float(mesh.bounds[1,2]-wall),8),
             degenerate_faces=int(np.count_nonzero(mesh.area_faces<1e-10)))
    return mesh,m


def slab_probe():
    # Two overlapping flat-roof prisms plus a cantilevered volume.
    parts=[(box(0,0,12,8),0,6),(box(8,0,16,8),0,12),(box(2,2,6,6),8,10)]
    levels=sorted({z for _,a,b in parts for z in (a,b)})
    layers=[]
    for lo,hi in zip(levels,levels[1:]):
        mid=(lo+hi)/2; p=s.union_all([g for g,a,b in parts if a<mid<b]);layers.append(p)
    vol=sum(p.area*(hi-lo) for p,lo,hi in zip(layers,levels,levels[1:]))
    empty=Polygon(); faces=[]
    for i,z in enumerate(levels):
        below=layers[i-1] if i else empty;above=layers[i] if i<len(layers) else empty
        faces.append({'z':z,'up_area_m2':below.difference(above).area,'down_area_m2':above.difference(below).area})
    return {'naive_sum_volume_m3':sum(g.area*(b-a) for g,a,b in parts),'union_volume_m3':vol,'horizontal_exposure':faces}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='geometry_probe_results.json');ap.add_argument('--models',default=None);args=ap.parse_args()
    out={'versions':{'shapely':s.__version__,'geos':s.geos_version_string,'trimesh':trimesh.__version__},'triangulation':{},'meshes':{},'repair':{},'slabs':slab_probe()}
    scene=trimesh.Scene(); index=0
    for name,raw in examples().items():
        if not raw.is_valid:
            repaired=s.make_valid(raw);out['repair'][name]={'type':repaired.geom_type,'polygon_count':len(list(polys(repaired))),'area_after':repaired.area};continue
        out['triangulation'][name]={'cdt':tri_metrics(raw,cdt(raw)), 'unconstrained_filtered':tri_metrics(raw,[t for t in triangulate(raw) if raw.covers(t)])}
        for mode in ['flat','gable','hip_convex']:
            if mode=='hip_convex' and (raw.interiors or raw.convex_hull.symmetric_difference(raw).area>1e-9 or name=='collinear'):continue
            mesh,m=mesh_build(raw,mode);out['meshes'][name+'_'+mode]=m
            if args.models:
                mesh.apply_translation([(index%5)*26,(index//5)*20,0])
                # Render-only copy: split vertices for hard face colors/normals.
                mesh.unmerge_vertices()
                mesh.visual.vertex_colors=np.repeat(mesh.visual.face_colors,3,axis=0)
                scene.add_geometry(mesh,node_name=name+'_'+mode,geom_name=name+'_'+mode);index+=1
    rng=random.Random(7321); cases=[]
    for _ in range(250):
        n=rng.randint(6,20); points=[]
        for i in range(n):
            a=2*math.pi*(i+rng.uniform(-.3,.3))/n;r=rng.uniform(1,12);points.append((r*math.cos(a),r*math.sin(a)))
        p=Polygon(points)
        if p.is_valid:cases.append(p)
    summary={}
    for method in ['cdt','unconstrained_filtered']:
        start=time.perf_counter(); errors=[]; counts=[]
        for p in cases:
            tris=cdt(p) if method=='cdt' else [t for t in triangulate(p) if p.covers(t)]
            m=tri_metrics(p,tris);errors.append(m['missing_m2']);counts.append(len(tris))
        summary[method]={'cases':len(cases),'nonzero_missing_cases':sum(e>1e-7 for e in errors),'max_missing_m2':max(errors),'total_triangles':sum(counts),'elapsed_s_including_validation':round(time.perf_counter()-start,4)}
    out['random_triangulation']=summary
    out['mesh_summary']={'cases':len(out['meshes']),'watertight':sum(m['watertight'] for m in out['meshes'].values()),'consistent_winding':sum(m['winding_consistent'] for m in out['meshes'].values()),'positive_volume':sum(m['positive_volume'] for m in out['meshes'].values()),'max_volume_error_m3':max(m['volume_error_m3'] for m in out['meshes'].values())}
    Path(args.output).write_text(json.dumps(out,indent=2))
    if args.models:
        # glTF convention: right-handed +Y up; core is +Z up.
        scene.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2,[1,0,0]))
        Path(args.models).write_bytes(scene.export(file_type='glb'))
    print(json.dumps({'random':summary,'mesh_summary':out['mesh_summary'],'slabs':out['slabs'],'repair':out['repair']},indent=2))

if __name__=='__main__':main()
