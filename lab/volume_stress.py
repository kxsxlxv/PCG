"""1024 distinct IDs, 8 rule families, real host + detail triangle planning.
Not a 1024-building GLB export or GPU benchmark.
"""
from dataclasses import asdict
from pathlib import Path
import json
from pcg.footprint import Footprint
from pcg.reference_grammar import build
from zero_reference_grammar import fixtures, swept_pipe
import unified_detail as detail

def run():
    base=[s for variant in (0,1) for s in fixtures(variant)]
    rows=[]
    for j in range(64):
        for s in base:
            src=s['footprint'];bid=f'volume31:{j}:{s["id"]}'
            scale=1.+.005*(j%11)
            f=Footprint(bid,[(x*scale,y*scale) for x,y in src.exterior],[[(x*scale,y*scale) for x,y in ring] for ring in src.holes])
            kw=dict(roof=s['roof'],eave=s['eave'],party_edges=s['party'],volume=True)
            v,p=build(f,s['rule'],s['floors'],s['top'],**kw)
            assert v.plan.footprint_digest==f.digest() and v.top_m==s['top']
            rec=dict(id=bid,offset=[0.,0.,0.],footprint=asdict(f),decision=v.decision(),architecture=asdict(v.plan))
            q=detail.plan(rec,swept_pipe,(f,v),p)
            assert 0<q['lite']<=q['near']<=1800
            assert all(not w.openings for w in v.plan.facades if w.party)
            rows.append(dict(id=bid,near=q['near'],lite=q['lite'],modules=len(q['modules']),projections=sum(m.kind=='ref_projection' for m in q['modules'])))
    assert len(rows)==len({r['id'] for r in rows})==1024
    report=dict(buildings=len(rows),near_max=max(r['near'] for r in rows),near_total=sum(r['near'] for r in rows),lite_total=sum(r['lite'] for r in rows),scope='actual detail mesh planning; no GLB export in stress test',rows=rows)
    Path('artifacts/zero-volume-grammar/stress.json').write_text(json.dumps(report,indent=2))
    print('STAGE31_STRESS',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':run()
