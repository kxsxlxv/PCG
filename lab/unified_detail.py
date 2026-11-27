"""Stage 26: metric details on the existing canonical archetype plan.
No GIS mutation. Use the existing massing, box and swept-pipe primitives.
Geometry is building-local; scene export applies the explicit tile offset.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import trimesh
from shapely import constrained_delaunay_triangles, set_precision
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.polygon import orient
from pcg.footprint import Footprint
from pcg.massing import build_massing
from pcg.geometry import build_shell
from pcg.contracts import Rectangle
from pcg.archetypes import plan_variant, stable
from pcg.architecture import BALCONY_DEPTH, BALCONY_THRESHOLD, BALCONY_WIDTH_EXTRA
from pcg._grammar_boxes import _oriented_box

EPS=1e-8
CAP=1800
EXT=.32


def polygons(g):
    if g.geom_type=='Polygon':
        if g.area>EPS:yield orient(g,1.)
    elif hasattr(g,'geoms'):
        for p in g.geoms:yield from polygons(p)


def restore(record):
    d=record['decision'];a=record['architecture'];f=Footprint(**record['footprint'])
    v=plan_variant(f,d['archetype']['use'],len(a['floor_levels'])-1,d['top_m'],
        region=d['region'],archetype=d['archetype']['id'] if d['selection']=='explicit' else None,
        max_triangles=d['max_triangles'],**{k:tuple(map(tuple,x)) if k=='party_edges' else x for k,x in record['source_constraints'].items()})
    assert v.plan.digest()==d['plan_digest'] and v.mass.actual_roof==d['actual_roof']
    return f,v


def roof_planes(mass):
    s=mass.shell;roles={x.id:x.role for x in s.surfaces};v=np.array(s.vertices)
    return [(Polygon(t[:,:2]),np.linalg.solve(np.c_[t[:,:2],np.ones(3)],t[:,2]))
            for k,inds in s.triangles() if roles[k]=='roof' for t in (v[list(inds)],)]


def height_at(planes,x,y):
    q=Point(x,y);h=[float(A@[x,y,1.]) for p,A in planes if p.buffer(1e-7).covers(q)]
    assert h and max(h)-min(h)<2e-6,(x,y,h)
    return sum(h)/len(h)


def support(planes,poly):
    zz=[];area=0.
    for p,A in planes:
        for q in polygons(p.intersection(poly)):
            area+=q.area;zz.extend(float(A@[x,y,1.]) for x,y in q.exterior.coords)
    assert zz and abs(area-poly.area)<1e-6,(area,poly.area)
    return min(zz),max(zz)


def solid(poly,A,bottom_delta,top_delta):
    """Closed vertical extrusion of a polygon on an affine roof plane.
    CDT caps retain holes, side winding uses the oriented boundary rings.
    """
    poly=orient(set_precision(poly,1e-7),1.);out=[]
    assert poly.geom_type=="Polygon" and not poly.is_empty
    def p(xy,d):return (*xy,float(A@[*xy,1.])+d)
    caps=list(constrained_delaunay_triangles(poly).geoms)
    assert abs(sum(t.area for t in caps)-poly.area)<1e-7
    for t in caps:
        xy=list(orient(t,1.).exterior.coords)[:-1]
        out.extend(([p(q,top_delta) for q in xy],[p(q,bottom_delta) for q in xy[::-1]]))
    for r in (poly.exterior,*poly.interiors):
        for a,b in zip(r.coords,list(r.coords)[1:]):
            q=[p(a,bottom_delta),p(b,bottom_delta),p(b,top_delta),p(a,top_delta)]
            out.extend(([q[i] for i in (0,1,2)],[q[i] for i in (0,2,3)]))
    t=np.array(out,float);m=trimesh.Trimesh(vertices=t.reshape(-1,3),faces=np.arange(t.size//3).reshape(-1,3),process=True)
    assert m.is_watertight and m.is_winding_consistent and m.volume>EPS
    assert abs(m.volume-poly.area*(top_delta-bottom_delta))<1e-6
    return t


def box_mesh(center,tangent,outward,w,d,h,z):
    v,faces=_oriented_box(tuple(center),tuple(tangent),tuple(outward),w,d,h,z)
    return np.array([[v[i] for i in (f[0],f[j],f[j+1])] for f in faces for j in range(1,len(f)-1)])


def wall_box(a,u,U,d,D,z,Z):
    t=np.array(a.end)-a.start;t/=a.length
    n=np.array(a.outward);p=np.array(a.start)+(u+U)/2*t+(d+D)/2*n
    return box_mesh(p,t,n,U-u,D-d,Z-z,z)


def quad(a,b,z,Z):
    q=np.array([[*a,z],[*b,z],[*b,Z],[*a,Z]])
    return q[[[0,1,2],[0,2,3]]]


def extent(parts):
    p=np.concatenate([t.reshape(-1,3) for _,_,t in parts]);return p.min(0),p.max(0)


def intersects(a,b,clear=0.):
    return min(a[2],b[2]+clear)>max(a[0],b[0]-clear)+EPS and min(a[3],b[3]+clear)>max(a[1],b[1]-clear)+EPS


@dataclass
class Detail:
    id: str
    kind: str
    geometry: list=field(default_factory=list)
    cuts: list=field(default_factory=list)
    antenna: tuple|None=None
    primary: bool=True
    anchor: str|None=None

    @property
    def triangles(self):
        return sum(len(t) for _,_,t in self.geometry)+sum(2*(len(p) if closed else len(p)-1) for _,p,_,closed in self.cuts)+2*(self.antenna is not None)


def plan(record,pipe_fn,prepared=None,reference=None):
    f,v=prepared if prepared is not None else restore(record);a=v.archetype;poly=f.polygon();planes=roof_planes(v.mass);h=v.eave_m
    shell_n=len(v.mass.shell.triangles());mods=[];rejected=[];occup=[];pairs={};actual={};checks=0
    extension_drop=0.
    walls=v.plan.facades
    for w in walls:actual[w.id]={o.id:(o.u,o.z,o.u+o.width,o.z+o.height) for o in w.openings}
    if reference is not None:
        for e in reference.entries:actual[e.facade][e.id]=e.rect
    def env(w,u,U,d,D):
        return Polygon([w.point(x,0,-y)[:2] for x,y in ((u,d),(U,d),(U,D),(u,D))])
    def clash(p,z,Z,skip=None):
        nonlocal checks
        for key,q,l,H in occup:
            if key==skip:continue
            checks+=1
            if min(Z,H)>max(z,l)+EPS and p.intersection(q).area>EPS:return True
        return False
    def accept(m,p=None,z=None,Z=None,*,outside=False):
        if shell_n+sum(x.triangles for x in mods)+m.triangles>min(CAP,v.max_triangles):
            rejected.append((m.id,'complete_module_budget'));return False
        for _,_,t in m.geometry:
            assert np.isfinite(t).all() and np.all(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)>1e-9)
        if p is not None:
            if outside and p.intersection(poly).area>EPS:
                rejected.append((m.id,'host_intersection'));return False
            if clash(p,z,Z):rejected.append((m.id,'accessory_intersection'));return False
            occup.append((m.id,p,z,Z))
        mods.append(m);return True
    def mid(key):return f.building_id+':unified:'+key
    # Keep every accepted stage-25 anchor. Reserve original envelopes while
    # widening each one, so an earlier widening cannot evict a later balcony.
    balconies=[]
    for w in walls:
        for o in w.openings:
            if o.role!='balcony_door':continue
            assert not w.courtyard and not w.party and o.floor>0
            k=o.id+':balcony';c=o.u+o.width/2;z=o.z;floor=v.plan.floor_levels[o.floor];ceil=v.plan.floor_levels[o.floor+1]
            glazed=v.glazed(w.id,o.id);top=min(ceil-.025,z+o.height+.14) if glazed else z+.95
            assert not glazed or top-.075>z+o.height
            w0=o.width+BALCONY_WIDTH_EXTRA;p=env(w,c-w0/2,c+w0/2,0,BALCONY_DEPTH)
            assert p.intersection(poly).area<EPS
            occup.append((k,p,floor+.015,top));balconies.append((k,w,o,c,z,floor,top,glazed))
    for k,w,o,c,z,floor,top,glazed in balconies:
        chosen=None
        for width in dict.fromkeys((min(2.10,v.plan.profile.spacing-.20),max(1.40,o.width),o.width)):
            if width<o.width or width>v.plan.profile.spacing-.10:continue
            slab=width+BALCONY_WIDTH_EXTRA;u,U=c-slab/2,c+slab/2
            rect=(c-width/2,z,c+width/2,z+o.height)
            p=env(w,u,U,0,BALCONY_DEPTH)
            if u<.08 or U>w.length-.08 or p.intersection(poly).area>EPS or clash(p,floor+.015,top,k):continue
            if any(intersects(rect,q,.035) for key,q in actual[w.id].items() if key!=o.id):continue
            doorw=min(.80,width*.48);post=.085
            if width-doorw-post<.45:continue
            x,_,X,Y=rect;winz=floor+v.plan.profile.sill
            if winz>=Y-.2:continue
            if stable('pair-v1',k)%2:door=(X-doorw,z,X,Y);win=(x,winz,X-doorw-post,Y)
            else:door=(x,z,x+doorw,Y);win=(x+doorw+post,winz,X,Y)
            assert not intersects(door,win)
            chosen=(u,U,p,rect,door,win);break
        if chosen is None:
            slab=o.width+BALCONY_WIDTH_EXTRA;u,U=c-slab/2,c+slab/2;p=env(w,u,U,0,BALCONY_DEPTH)
            rejected.append((k,'unpaired_original_aperture'))
        else:
            u,U,p,rect,door,win=chosen;actual[w.id][o.id]=rect
            pairs[o.id]=dict(door=door,window=win,room_id=k,slab_top=z,width_m=rect[2]-rect[0])
        occup[:]=[q for q in occup if q[0]!=k]
        # 155 mm slab, threshold unchanged from the canonical opening.
        assert abs(z-floor-BALCONY_THRESHOLD)<EPS
        parts=[('slab','concrete_panel',wall_box(w,u,U,0,BALCONY_DEPTH,z-.155,z))]
        path=[w.point(x,z,-d) for x,d in ((u,0),(u,BALCONY_DEPTH),(U,BALCONY_DEPTH),(U,0))]
        m=Detail(k,'balcony_glazed' if glazed else 'balcony_open',parts,anchor=o.id)
        if glazed:
            cap=top-.075
            for A,B in zip(path,path[1:]):parts.append(('stage21_glazed','painted_rail',quad(A[:2],B[:2],z,cap)))
            parts.append(('stage21_cap','bitumen_roof',wall_box(w,u,U,0,BALCONY_DEPTH,cap,top)))
        else:m.cuts.append(('balcony',path,.95,False))
        assert accept(m,p,z-.155,top,outside=True),'canonical balcony cannot be removed'
    # Reference grammar: projected window bays are full-height attachments,
    # not internal loggias. Their front reuses the host opening record in the
    # research renderer. No host opening or shell vertex is removed.
    if reference is not None:
        bywall={w.id:w for w in walls}
        for q in reference.projections:
            w=bywall[q.facade];p=env(w,q.u,q.U,0,q.depth)
            if p.intersection(poly).area>EPS or clash(p,q.z,q.Z):
                rejected.append((q.id,'projection_clearance'));continue
            tris=wall_box(w,q.u,q.U,0,q.depth,q.z,q.Z)
            n=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
            outward=np.r_[w.outward,0.];front=n@outward>.99;back=n@outward<-.99
            assert front.sum()==2 and back.sum()==2
            parts=[('ref_projection_front',a.wall,tris[front]),('ref_projection_side',a.wall,tris[~(front|back)])]
            accept(Detail(q.id,'ref_projection',parts,primary=True),p,q.z,q.Z,outside=True)
    # Validate widened apertures against all other openings; sidecar is bounded
    # by its original regular-cell lookup interval (no overlapping slot search).
    for w in walls:
        rects=list(actual[w.id].values())
        for i,r in enumerate(rects):
            assert 0<r[0]<r[2]<w.length and 0<r[1]<r[3]<=h
            assert not any(intersects(r,q) for q in rects[:i])
    # Roof edge construction: flat closed parapet ring; pitched slope follows
    # the existing roof exactly via uniformly offset canonical massing.
    if v.mass.actual_roof=='flat':
        inner=poly.buffer(-.20,join_style=2);band=poly.difference(inner)
        if not inner.is_empty:
            A=np.array([0.,0.,h]);parts=[];cut=[]
            for q in polygons(band):
                parts.extend([('parapet','concrete_panel',solid(q,A,-.015,.42)),('coping','painted_trim',solid(q,A,.42,.465))])
            for q in polygons(poly.buffer(-.10,join_style=2)):
                for ring in (q.exterior,*q.interiors):cut.append(('roof',[(x,y,h+.465) for x,y in list(ring.coords)[:-1]],.86,True))
            accept(Detail(mid('parapet'),'parapet',parts,cut),band,h-.015,h+1.325)
        else:rejected.append((mid('parapet'),'insufficient_roof_width'))
    elif not any(w.party for w in walls):
        outer=poly.buffer(EXT,join_style=2);rise=v.top_m-h
        if v.mass.actual_roof=='gable':
            r=list(poly.minimum_rotated_rectangle.exterior.coords);run=min(math.dist(x,y) for x,y in zip(r,r[1:]))/2
        else:
            run=max(Point(x,y).distance(poly.boundary) for x,y,z in v.mass.shell.vertices if abs(z-v.top_m)<EPS)
        extra=rise/run*EXT;extension_drop=extra;expanded=Footprint(mid('overhang_shell'),list(outer.exterior.coords))
        mass=build_massing(expanded,h-extra,v.mass.actual_roof,rise+extra)
        assert not mass.fallback_reason
        new=roof_planes(mass)
        for p,A in planes:
            for x,y in (*p.exterior.coords,(p.centroid.x,p.centroid.y)):
                assert abs(height_at(new,x,y)-float(A@[x,y,1.]))<2e-6
        annulus=outer.difference(poly);parts=[];area=0.
        for p,A in new:
            for q in polygons(p.intersection(annulus)):
                area+=q.area;t0=solid(q,A,-.075,0.)
                top=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0])[:,2]>1e-9
                parts.extend([('eave_cover',a.pitched_material,t0[top]),('soffit','painted_trim',t0[~top])])
        assert abs(area-annulus.area)<1e-6
        lo,hi=extent(parts);accept(Detail(mid('overhang'),'overhang',parts),annulus,lo[2],hi[2],outside=True)
    else:rejected.append((mid('overhang'),'party_roof_edge'))
    if v.mass.actual_roof!='flat' and a.use=='apartments':
        for w in walls:
            if w.party or w.courtyard or w.length<3:continue
            if abs(height_at(planes,*w.point(w.length/2,0)[:2])-h)>1e-6:continue
            xy=[w.point(u,0,.24)[:2] for u in (.70,w.length-.70)]
            z=[height_at(planes,*p) for p in xy]
            if abs(z[0]-z[1])>1e-6 or abs(height_at(planes,*((np.array(xy[0])+xy[1])/2))-z[0])>1e-6:continue
            p=LineString(xy).buffer(.04,cap_style=2)
            if not poly.covers(p):continue
            accept(Detail(mid('roof_guard:'+w.id.rsplit(':facade:',1)[-1]),'roof_guard',cuts=[('roof',[(*q,z[0]-.015) for q in xy],.86,False)]),p,z[0]-.015,z[0]+.845)
    # Roof equipment fits the real roof planes and stays clear of the edge ring.
    # A stable candidate grid, aligned to the footprint's longest facade.
    t=np.array(max(walls,key=lambda w:w.length).end)-max(walls,key=lambda w:w.length).start;t/=np.linalg.norm(t);n=np.array([t[1],-t[0]])
    bx,by0,BX,BY=poly.bounds
    centers=[(bx+(BX-bx)*x,by0+(BY-by0)*y) for x in (.25,.5,.75,.36,.64) for y in (.25,.5,.75,.36,.64)]
    if reference is not None and v.mass.actual_roof=='flat':
        bywall={w.id:w for w in walls}
        for q in reference.roof_cores:
            w=bywall[q.facade];t0=(np.asarray(w.end)-w.start)/w.length;n0=np.asarray(w.outward)
            p=np.asarray(w.point(q.u,0,q.depth)[:2]);W,D=2.55,1.85
            area=Polygon([p+x*t0*W/2+y*n0*D/2 for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])
            if not poly.buffer(-.35).covers(area) or clash(area,h,h+1.55):
                rejected.append((q.id,'roof_core_clearance'));continue
            body=box_mesh(p,t0,n0,W,D,1.45,h);cap=box_mesh(p,t0,n0,W,D,.10,h+1.45)
            accept(Detail(q.id,'ref_roof_core',[('ref_roof_core',a.wall,body),('ref_roof_cap','bitumen_roof',cap)],primary=True),area,h,h+1.55)
    def equipment(kind,w,d,height,family):
        for x,y in sorted(centers,key=lambda q:stable('roof-place-v1',f.building_id,kind,*q)):
            p=np.array([x,y]);foot=Polygon([p+i*t*w/2+j*n*d/2 for i,j in ((-1,-1),(1,-1),(1,1),(-1,1))]);flash=foot.buffer(.10,join_style=2)
            if not poly.buffer(-.35).covers(flash):continue
            low,high=support(planes,foot);Z=high+height
            crown=.35 if kind=='dormer' else .16
            if clash(flash,low-.03,Z+crown):continue
            body=box_mesh(p,t,n,w,d,Z-low+.03,low-.03);parts=[]
            if kind in ('dormer','access'):
                # Front proxy is evaluated on actual dormer housing, not on
                # the sloped main roof. Orient front toward the descending slope.
                slope=np.array([A[:2] for q,A in planes if q.buffer(1e-7).covers(Point(x,y))][0]);direction=-slope if kind=='dormer' else n
                normals=np.cross(body[:,1]-body[:,0],body[:,2]-body[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
                vert=np.flatnonzero(abs(normals[:,2])<.1);j=vert[np.argmax(normals[vert,:2]@direction)];front=np.all(abs(normals-normals[j])<1e-7,axis=1)
                parts=[(kind+'_front',family,body[front]),(kind+'_body',family,body[~front])]
            else:parts=[(kind,family,body)]
            if kind in ('access','vent'):
                parts.append(('access_cover' if kind=='access' else 'equipment_cap','bitumen_roof' if v.mass.actual_roof=='flat' else a.pitched_material,box_mesh(p,t,n,w+.14,d+.14,.075,Z)))
            if kind=='dormer':
                # Actual small gabled cap, using the existing canonical rectangular shell.
                D,W=d+.16,w+.16
                cap=build_shell(Rectangle(mid('dormer_cap'),D,W,.075,'gable',.27))
                cv=np.array(cap.vertices);world=np.c_[p+(cv[:,0,None]-D/2)*n+(cv[:,1,None]-W/2)*t,cv[:,2]+Z]
                roles={s.id:s.role for s in cap.surfaces}
                for role in ('roof','wall','base'):
                    tris=np.array([world[list(i)] for sid,i in cap.triangles() if roles[sid]==role])
                    parts.append(('dormer_roof' if role=='roof' else 'dormer_trim',a.pitched_material if role=='roof' else 'painted_trim',tris))
            if kind=='chimney':
                ring=Point(x,y).buffer(min(w,d)*.24,quad_segs=2).difference(Point(x,y).buffer(min(w,d)*.16,quad_segs=2))
                for q in polygons(ring):parts.append(('flue','painted_rail',solid(q,np.array([0.,0.,Z]),0.,.16)))
            for q,A in planes:
                for b in polygons(q.intersection(flash.difference(foot))):
                    for tri in constrained_delaunay_triangles(b).geoms:
                        pts=list(orient(tri,1.).exterior.coords)[:-1]
                        parts.append(('flashing','painted_rail',np.array([[(*p,float(A@[*p,1.])+.014) for p in pts]])))
            m=Detail(mid(kind),kind,parts,primary=kind!='vent')
            return accept(m,flash,low-.03,Z+crown)
        rejected.append((mid(kind),'no_supported_clearance'));return False
    if v.mass.actual_roof=='flat':
        if a.use!='house':equipment('access',2.4,1.9,1.65,'plaster_cream');equipment('vent',.9,.85,.7,'concrete_panel')
        else:equipment('vent',.7,.65,.6,'concrete_panel')
    else:
        equipment('chimney',.65,.72,.90,'red_brick')
        if a.use!='office':equipment('dormer',1.25,1.05,.82,'plaster_cream')
    if v.mass.actual_roof=='flat':
        for x,y in centers:
            q=Point(x,y).buffer(.19,quad_segs=3)
            if not poly.buffer(-.35).covers(q) or clash(q,h,h+.025):continue
            if accept(Detail(mid('roof_drain'),'roof_drain', [('roof24_drain','painted_rail',solid(q,np.array([0.,0.,h]),0.,.025))],primary=False),q,h,h+.025):break
    # Antenna card: same shared atlas as earlier stage, supported on actual roof.
    if a.use!='office':
        for x,y in centers:
            q=Point(x,y).buffer(1.)
            if not poly.buffer(-.25).covers(q):continue
            z=height_at(planes,x,y)
            if clash(q,z,z+2.2):continue
            k=int(stable('antenna-v1',f.building_id)%4)
            if accept(Detail(mid('antenna'),'antenna',antenna=((x,y,z),(*t,0.),k)),q,z,z+2.2):break
    # Exterior detail placement uses the widened opening envelope and real
    # obstacle polygons; all horizontal/vertical modules stay off apertures.
    def wall_add(w,key,kind,u,U,d,D,z,Z,mat,primary=False):
        if not (.06<=u<U<=w.length-.06 and 0<=z<Z<=h):return False
        if any(intersects((u,z,U,Z),r,0. if kind=='landing' else .025) for r in actual[w.id].values()):return False
        p=env(w,u,U,d,D)
        if p.intersection(poly).area>EPS or clash(p,z,Z):return False
        return accept(Detail(mid(w.id.rsplit(':facade:',1)[-1]+':'+key),kind,[(kind,mat,wall_box(w,u,U,d,D,z,Z))],primary=primary),p,z,Z,outside=True)
    for w in walls:
        if w.party or w.courtyard:continue
        door=next((o for o in w.openings if o.role=='door'),None)
        entries=[('',(door.u,door.z,door.u+door.width,door.z+door.height))] if door else []
        if reference is not None:entries.extend((f':section:{e.bay}',e.rect) for e in reference.entries if e.facade==w.id)
        for suffix,(u,bottom,U,z) in entries:
            # The landing ends exactly at the source threshold; canopy clears all openings.
            wall_add(w,'landing'+suffix,'landing',u-.15,U+.15,0,.70,0,bottom,'stone_plinth',True)
            for zz in (z+.12,z+.22):
                if wall_add(w,'canopy'+suffix,'canopy',u-.30,U+.30,0,.85,zz,zz+.12,'painted_trim',True):break
        if a.use=='apartments' and w.length>10:
            for o in w.openings:
                if o.role=='window' and o.floor>0 and stable('ac-v1',o.id)%11==0:
                    c=o.u+o.width/2
                    if wall_add(w,'ac:'+o.id.rsplit(':',2)[-2]+':'+o.id.rsplit(':',1)[-1],'ac',c-.34,c+.34,0,.29,o.z-.49,o.z-.08,'painted_trim'):break
        # Open troughs along actual pitched eaves (not gable ends). Downpipe
        # uses the earlier continuous six-sided sweep and fits a free pier.
        if v.mass.actual_roof!='flat' and not w.courtyard:
            p=w.point(w.length/2,0);at=height_at(planes,*p[:2]);
            if abs(at-h)<1e-6 and w.length>2:
                z=h-extension_drop-.17
                u,U=.10,w.length-.10
                parts=[('gutter_floor','painted_rail',wall_box(w,u,U,.33,.51,z,z+.025))]
                parts.extend(('gutter_side','painted_rail',wall_box(w,u,U,d,d+.025,z+.025,z+.13)) for d in (.33,.485))
                p=env(w,u,U,.33,.51);key=mid(w.id.rsplit(':facade:',1)[-1]+':gutter')
                if z>.9 and accept(Detail(key,'gutter',parts,primary=False),p,z,z+.13,outside=True):
                    tangent=(np.array(w.end)-w.start)/w.length;out=np.array(w.outward);origin=np.r_[w.start,0.]
                    for u in (.30,w.length-.30):
                        lo,hi=.14,z
                        if any(intersects((u-.065,lo,u+.065,hi),r,.025) for r in actual[w.id].values()):continue
                        pp=env(w,u-.05,u+.05,.025,.465)
                        if pp.intersection(poly).area>EPS or clash(pp,lo,hi):continue
                        path=[(u,.085,lo),(u,.085,z-.48),(u,.42,z-.14),(u,.42,z)]
                        tris=pipe_fn(path,radius=.045,n=6);T=np.column_stack((np.r_[tangent,0.],np.r_[out,0.],[0,0,1.]));tris=tris@T.T+origin
                        if np.linalg.det(T)<0:tris=tris[:,::-1]
                        if accept(Detail(mid(w.id.rsplit(':facade:',1)[-1]+':drain'),'drain',[('bent_drain','painted_rail',tris)],primary=False),pp,lo,hi,outside=True):break
        # Segment plinth and optional belt around all apertures and balcony
        # obstacles. Avoid corner miters on arbitrary GIS polygons.
        bands=[('plinth',.12,.22,.075,'stone_plinth')] if a.id=='glass_grid' else [('plinth',.43,.53,.075,'stone_plinth')]
        if a.composition in ('sections','paired','blind_end') and len(v.plan.floor_levels)>2:
            Z=v.plan.floor_levels[1]-.17;bands.append(('belt',Z-.09,Z,.09,'painted_trim'))
        for kind,z,Z,d,mat in bands:
            if not 0<z<Z<h:continue
            spans=[(.12,w.length-.12)]
            for r in actual[w.id].values():
                if min(Z,r[3]+.025)>max(z,r[1]-.025):
                    spans=[q for u,U in spans for q in ((u,min(U,r[0]-.035)),(max(u,r[2]+.035),U)) if q[1]-q[0]>.12]
            for i,(u,U) in enumerate(spans):wall_add(w,kind+str(i),kind,u,U,0,d,z,Z,mat)
    assert len({m.id for m in mods})==len(mods)
    off=np.asarray(record['offset'],float)
    for m in mods:
        for _,_,t0 in m.geometry:
            q=(t0+off).astype(np.float32).astype(float)
            assert np.all(np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)>1e-9),(m.id,'float32 geometry collapse')
    assert {m.anchor for m in mods if m.anchor}=={o.id for _,_,o,*_ in balconies}
    total=shell_n+sum(m.triangles for m in mods);lite=shell_n+sum(m.triangles for m in mods if m.primary)
    assert 0<lite<=total<=min(CAP,v.max_triangles)
    return dict(variant=v,footprint=f,modules=mods,pairs=pairs,rejected=rejected,checks=checks,near=total,lite=lite,shell=shell_n)
