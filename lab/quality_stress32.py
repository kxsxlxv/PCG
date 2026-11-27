"""Actual compact geometry for 1024 IDs; matched base plans and anchors."""
from dataclasses import asdict
from pathlib import Path
import json
from pcg.footprint import Footprint
from pcg.reference_grammar import build
from zero_reference_grammar import fixtures,swept_pipe
import unified_detail as detail

def run():
    base=[s for variant in (0,1) for s in fixtures(variant)];rows=[]
    for j in range(64):
        for s in base:
            src=s['footprint'];bid=f'quality32:{j}:{s["id"]}';scale=1.+.005*(j%11)
            f=Footprint(bid,[(x*scale,y*scale) for x,y in src.exterior],[[(x*scale,y*scale) for x,y in ring] for ring in src.holes])
            v,p=build(f,s['rule'],s['floors'],s['top'],roof=s['roof'],eave=s['eave'],party_edges=s['party'],volume=True)
            rec=dict(id=bid,offset=[0.,0.,0.],footprint=asdict(f),decision=v.decision(),architecture=asdict(v.plan))
            old=detail.plan(rec,swept_pipe,(f,v),p);new=detail.plan(rec,swept_pipe,(f,v),p,efficient=True)
            assert v.plan.footprint_digest==f.digest() and v.top_m==s['top']
            assert [(m.id,m.anchor,m.kind,m.primary) for m in old['modules']]==[(m.id,m.anchor,m.kind,m.primary) for m in new['modules']]
            assert old['pairs']==new['pairs'] and old['rejected']==new['rejected']
            assert old['near']-new['near']==new['compact_saved'] and 0<new['lite']<=new['near']<=old['near']<=1800
            rows.append(dict(id=bid,near=new['near'],lite=new['lite'],old_near=old['near'],saved=new['compact_saved']))
    assert len(rows)==len({r['id'] for r in rows})==1024
    report=dict(buildings=len(rows),near_max=max(r['near'] for r in rows),near_total=sum(r['near'] for r in rows),lite_total=sum(r['lite'] for r in rows),saved=sum(r['saved'] for r in rows),scope='matched actual base detail geometry plans before Stage28/29 refinements',rows=rows)
    root=Path('artifacts/zero-efficient-grammar');root.mkdir(parents=True,exist_ok=True)
    (root/'stress.json').write_text(json.dumps(report,indent=2))
    print('STAGE32_STRESS',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':run()
