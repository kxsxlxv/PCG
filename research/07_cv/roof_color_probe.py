"""CPU-only roof appearance observation, synthetic regression; no ML or fitting."""
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np


def observe(rgb, mask, gsd, alignment_uncertainty_m, invalid=None):
    """Return apparent Lab color, not calibrated albedo/material. Mask in image CRS."""
    if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
        raise ValueError('expected RGB uint8 and aligned 2D mask')
    if not np.isfinite([gsd, alignment_uncertainty_m]).all() or gsd <= 0 or alignment_uncertainty_m < 0:
        raise ValueError('invalid GSD or uncertainty')
    mask = mask.astype(bool)
    radius = max(1, int(np.ceil(alignment_uncertainty_m / gsd)))
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    core = dist > radius
    valid = core if invalid is None else core & ~invalid.astype(bool)
    n = int(valid.sum()); frac = n / max(1, int(mask.sum()))
    if n < 24 or frac < .25:
        return dict(status='abstain', reason='insufficient_safe_pixels', pixels=n, fraction=frac)
    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2Lab)[valid]
    center = np.median(lab, axis=0)
    delta = np.linalg.norm(lab-center, axis=1)
    spread = float(np.quantile(delta, .75))
    if spread > 18:
        return dict(status='abstain', reason='heterogeneous_appearance', pixels=n, fraction=frac, spread=spread)
    return dict(status='observed', apparent_lab=center.tolist(), pixels=n, fraction=frac, spread=spread)


def run():
    cv2.setNumThreads(1)
    rng=np.random.default_rng(20261124)
    # Representative synthetic colors, not a regional palette or measurements.
    palette=np.array([[165,77,58],[80,110,77],[105,108,113],[210,205,190],[57,61,65],[110,79,60]],np.uint8)
    errors=[]; naive=[]; rejects=0; cases=0; elapsed=0.; shift_rejects=0
    for i in range(600):
        w,h=rng.integers(26,65,2); x,y=15,15
        mask=np.zeros((96,96),np.uint8); mask[y:y+h,x:x+w]=1
        rgb=np.full((96,96,3),[55,110,45],np.uint8)
        base=palette[i%len(palette)]; clean=np.clip(base.astype(float)+rng.normal(0,4,(h,w,3)),0,255).astype(np.uint8)
        rgb[y:y+h,x:x+w]=clean
        # Contaminated perimeter plus marked shadow/vegetation patch.
        rgb[y:y+3,x:x+w]=[55,110,45];rgb[y:y+h,x:x+3]=[100,100,100]
        rgb[y+h-3:y+h,x:x+w]=[55,110,45];rgb[y:y+h,x+w-3:x+w]=[100,100,100]
        invalid=np.zeros_like(mask,bool); invalid[y+5:y+h//2,x+5:x+w//2]=True
        rgb[invalid]=[29,30,27]
        target=cv2.cvtColor(clean.astype(np.float32)/255,cv2.COLOR_RGB2Lab).reshape(-1,3).mean(0)
        all_lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)
        naive.append(float(np.linalg.norm(all_lab[mask>0].mean(0)-target)))
        start=time.perf_counter(); obs=observe(rgb,mask,.3,.9,invalid); elapsed+=time.perf_counter()-start
        assert obs==observe(rgb,mask,.3,.9,invalid)
        cases+=1
        if obs['status']=='observed': errors.append(float(np.linalg.norm(np.array(obs['apparent_lab'])-target)))
        else: rejects+=1
        # Reported high alignment uncertainty must force abstention, rather than moving footprint.
        shift_rejects += observe(rgb,mask,.3,15,invalid)['status']=='abstain'
    assert len(errors)==600 and shift_rejects==600
    # Heterogeneous test: half red, half white, neither single color authoritative.
    rgb=np.full((64,64,3),[165,77,58],np.uint8);rgb[:,32:]=[210,205,190]
    mixed=observe(rgb,np.ones((64,64),bool),.3,.3)
    assert mixed['status']=='abstain' and mixed['reason']=='heterogeneous_appearance'
    return dict(cases=cases,accepted=len(errors),rejected=rejects,high_uncertainty_abstentions=int(shift_rejects),mixed_case=mixed,naive_mean_delta_e76=float(np.mean(naive)),robust_mean_delta_e76=float(np.mean(errors)),robust_p95_delta_e76=float(np.quantile(errors,.95)),cpu_seconds=elapsed,mean_ms=elapsed/cases*1000,opencv=cv2.__version__,numpy=np.__version__,limitations=['synthetic RGB chips and known masks','invalid mask supplied by fixture, not automatically detected','no real aerial imagery or albedo recovery','no GPU or model inference','timing excludes I/O, geospatial raster access and fixture generation'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='roof_color_results.json');a=p.parse_args()
    r=run();Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
