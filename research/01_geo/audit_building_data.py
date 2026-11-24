#!/usr/bin/env python3
"""Research-only offline audit of WGS84 GeoJSON building layers. Requires shapely>=2, pyproj.
Does not conflate or change geometry. Counts features, not verified physical buildings.
"""
import argparse, json, math, re
from collections import Counter
from pathlib import Path
from statistics import median
from pyproj import CRS, Transformer
from shapely.geometry import shape, box
from shapely.ops import transform
from shapely.strtree import STRtree

FIELDS = {
    'height': ('height',), 'floors': ('num_floors', 'building:levels'),
    'roof_shape': ('roof_shape', 'roof:shape'), 'roof_height': ('roof_height','roof:height'),
    'roof_color': ('roof_color','roof:colour'), 'roof_material': ('roof_material','roof:material'),
    'facade_color': ('facade_color','building:colour'),
    'facade_material': ('facade_material','building:material'),
    'use': ('class','subtype','building:use','building'), 'year': ('start_date','building:age')}
MISSING = {'', 'unknown', 'null', 'none', '-1', 'nan', 'n/a'}

def present(v):
    return v is not None and str(v).strip().lower() not in MISSING

def field(p, keys):
    return next((p[k] for k in keys if k in p and present(p[k])), None)

def number(v, units=False):
    if not present(v) or isinstance(v, bool): return None
    s = str(v).strip().replace(',', '.')
    if units: s = re.sub(r'\s*m$', '', s, flags=re.I)
    try: n = float(s)
    except (ValueError, TypeError): return None
    return n if math.isfinite(n) and n > 0 else None

def audit(path, kind, bounds, project):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('type') != 'FeatureCollection': raise ValueError(f'{path}: expected GeoJSON FeatureCollection')
    counts=Counter(); coverage=Counter(); valid_numeric=Counter(); values={'height':[], 'floors':[]}; records=[]; ids=Counter(); sources=Counter()
    area=box(*bounds)
    for i,f in enumerate(data['features']):
        counts['input']+=1
        p=f.get('properties') or {}
        if kind=='osm' and not (present(p.get('building')) or present(p.get('building:part'))):
            counts['nonbuilding']+=1; continue
        if kind=='osm' and (p.get('building')=='no' and not present(p.get('building:part'))):
            counts['nonbuilding']+=1; continue
        if not f.get('geometry'):
            counts['missing_geometry']+=1; continue
        try: g=shape(f['geometry'])
        except Exception:
            counts['unreadable_geometry']+=1; continue
        if g.geom_type not in ('Polygon','MultiPolygon') or g.is_empty:
            counts['nonpolygon_or_empty']+=1; continue
        if any(not math.isfinite(x) for x in g.bounds) or not (-180<=g.bounds[0]<=g.bounds[2]<=180 and -90<=g.bounds[1]<=g.bounds[3]<=90):
            raise ValueError(f'{path}: non-WGS84 coordinates at feature {i}')
        if not g.is_valid:
            counts['invalid_geometry_unlocated']+=1; continue
        if not area.covers(g.representative_point()):
            counts['outside_aoi_owner']+=1; continue
        if kind=='osm' and present(p.get('building:part')) and p.get('building:part')!='no' or kind=='part':
            counts['valid_parts']+=1; continue
        if p.get('is_underground') is True:
            counts['underground']+=1; continue
        counts['valid_building_features']+=1
        if not area.covers(g): counts['crosses_aoi']+=1
        polygons = list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
        if len(polygons)>1: counts['multipart']+=1
        if any(len(poly.interiors)>0 for poly in polygons): counts['with_holes']+=1
        fid=p.get('id') or f.get('id') or (f"{p.get('@type')}/{p.get('@id')}" if '@id' in p else f'row:{i}')
        ids[str(fid)]+=1
        for k,keys in FIELDS.items():
            v=field(p,keys)
            if k=='use' and str(v).lower() in ('yes','building','unknown'): v=None
            if present(v): coverage[k]+=1
            if k in values:
                n=number(v, units=k=='height')
                if n is not None and (k!='floors' or n.is_integer()): valid_numeric[k]+=1; values[k].append(n)
        s=p.get('sources') or []
        if isinstance(s,list) and s and isinstance(s[0],dict): sources[str(s[0].get('dataset','unknown'))]+=1
        records.append((str(fid),transform(project,g)))
    n=counts['valid_building_features']
    report={'file':str(path),'kind':kind,'counts':dict(counts),'duplicate_id_excess':sum(c-1 for c in ids.values() if c>1),
      'present_fields':{k:{'n':coverage[k],'pct':round(100*coverage[k]/n,2) if n else None} for k in FIELDS},
      'positive_numeric_fields':{k:{'n':valid_numeric[k],'median':median(values[k]) if values[k] else None} for k in values},
      'primary_source_counts':dict(sources)}
    return report, records

def match(a,b):
    if not a or not b: return {'status':'one or both layers empty; no comparison'}
    tree=STRtree([g for _,g in b]); adj_a=Counter(); adj_b=Counter(); high_a=Counter(); high_b=Counter(); edges=[]; touches=0
    for i,(aid,g) in enumerate(a):
        for j0 in tree.query(g,predicate='intersects'):
            j=int(j0); bid,h=b[j]; inter=g.intersection(h).area
            if inter<=0: touches+=1; continue
            iou=inter/(g.area+h.area-inter); containment=inter/min(g.area,h.area)
            if iou>=.5 or containment>=.8 or (iou>=.3 and (g.covers(h.centroid) or h.covers(g.centroid))):
                high=iou>=.5; adj_a[i]+=1; adj_b[j]+=1
                if high: high_a[i]+=1; high_b[j]+=1
                edges.append((i,j,high))
    clean=sum(high and adj_a[i]==adj_b[j]==1 for i,j,high in edges)
    return {'candidate_pairs':len(edges),'high_iou_pairs':sum(e[2] for e in edges),'clean_high_1to1_pairs':clean,
       'a_no_candidate':len(a)-len(adj_a),'b_no_candidate':len(b)-len(adj_b),
       'a_multiple_candidates':sum(v>1 for v in adj_a.values()),'b_multiple_candidates':sum(v>1 for v in adj_b.values()),
       'boundary_touch_pairs':touches,'note':'Candidate matches only; unmatched is not proof of a missing building.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bbox',nargs=4,type=float,default=[39.91,43.42,39.94,43.44],metavar=('W','S','E','N'))
    ap.add_argument('--osm'); ap.add_argument('--overture'); ap.add_argument('--overture-parts'); ap.add_argument('--output',default='audit.json')
    args=ap.parse_args(); w,s,e,n=args.bbox
    if not (-180<=w<e<=180 and -90<s<n<90): ap.error('Invalid bbox')
    if not any([args.osm,args.overture,args.overture_parts]): ap.error('Provide at least one input file')
    local=CRS.from_proj4(f'+proj=laea +lat_0={(s+n)/2} +lon_0={(w+e)/2} +datum=WGS84 +units=m +no_defs')
    project=Transformer.from_crs('EPSG:4326',local,always_xy=True).transform
    result={'bbox_wsen':args.bbox,'area_rule':'full valid polygon with representative point inside bbox; no clipping','layers':{},'warnings':['Input snapshots must be recorded externally. Invalid geometry counted before AOI filter. Fields reported as present, not necessarily correct.']}
    rec={}
    for kind,p in [('osm',args.osm),('overture',args.overture),('part',args.overture_parts)]:
        if p: result['layers'][kind],rec[kind]=audit(p,kind,args.bbox,project)
    if 'osm' in rec and 'overture' in rec: result['matching']=match(rec['osm'],rec['overture'])
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(args.output)
if __name__=='__main__': main()
