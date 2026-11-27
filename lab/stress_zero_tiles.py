"""Actual 1024-building export and independent package integrity exercise."""
from pathlib import Path
import json
import resource
import time
from compile_zero_tiles import fixture,build_dataset,dump,regression_float32_boundary
from verify_zero_bundle import verify

def main():
    regression=regression_float32_boundary()
    print("ZERO_ABI_BOUNDARY_REGRESSION "+json.dumps(regression),flush=True)
    root=Path("artifacts").resolve()
    library=root/"zero-multitile/library/manifest.json"
    decoder=Path("/tmp/pcg_zero_tile_abi")
    source=fixture();base=source["buildings"]
    buildings=[]
    for i in range(1024):
        b=dict(base[i%len(base)])
        floors=(2+i%11) if b["profile"]!="house" else (1+i%3)
        b.update(id=f"stress:{i:05d}",anchor=[(i%32)*20.-325.,(i//32)*20.-325.,(i%7)*.125],
                 floors=floors,eave_m=floors*3.)
        buildings.append(b)
    source.update(buildings=buildings,tile_size_m=256)
    out=root/"zero-stress";out.mkdir(exist_ok=True);dump(out/"input.json",source)
    start=time.perf_counter()
    result,geometry,semantic=build_dataset(source,out/"tiles",library,decoder,audit_samples=64,preview=False)
    seconds=time.perf_counter()-start
    assert len(geometry)==result["buildings"]==1024
    assert result["cpp_abi_tiles"]==result["tiles"] and result["opening_triangles_added"]==0
    checked={}
    for key,p in (("96_128",root/"zero-multitile/tiles128/index.json"),
                  ("96_256",root/"zero-multitile/tiles256/index.json"),
                  ("approved_pilot",root/"zero-approved-pilot/tiles/index.json"),
                  ("1024_256",out/"tiles/index.json")):
        checked[key]=verify(p,root)
    # Damage an actual exported file, require verifier rejection, restore it.
    index=out/"tiles/index.json";d=json.loads(index.read_bytes())
    target=(index.parent/d["tiles"][0]["manifest"]).parent/"facade_params.bin"
    raw=target.read_bytes()
    try:
        target.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        try:verify(index,root)
        except ValueError as e:assert "SHA256 mismatch" in str(e)
        else:raise AssertionError("corrupted package accepted")
    finally:target.write_bytes(raw)
    assert verify(index,root)==checked["1024_256"]
    # Index path traversal must fail even for a readable file.
    bad=out/"tiles/path_escape_probe.json";d["library"]="/etc/hosts";dump(bad,d)
    try:
        try:verify(bad,root)
        except (ValueError,FileNotFoundError):pass
        else:raise AssertionError("external path accepted")
    finally:bad.unlink()
    summary={"float32_boundary_regression":regression,"actual_export":result,"compile_and_audit_seconds":seconds,
             "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "independent_verified_packages":checked,"rejected_corrupt_payload":True,"rejected_external_path":True,
             "limitations":["synthetic CPU export, not runtime loading or hardware GPU benchmark",
               "peak RSS is whole Linux process, not per-tile memory or VRAM",
               "debug JSON included in bundle sizes; no native mesh compression"]}
    dump(out/"report.json",summary);print("ZERO_STRESS_REPORT "+json.dumps(summary),flush=True)

if __name__=="__main__":main()
