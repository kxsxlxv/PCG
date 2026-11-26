"""Stage 20: external balcony only. Historic recess experiments are baseline only."""
from pathlib import Path
from collections import defaultdict, Counter
import base64, hashlib, io, json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
from pcg.materials import MATERIALS, texture_maps
from pcg.far_emission import srgb_decode
from compile_zero_tiles import pack_table, LAYERS
import compile_zero_geometry_tile as tile
import render_zero_geometry_facades as ref
import zero_facade_architecture as arch
import zero_roof_budgets as roof
import zero_combined_lod as combined
import zero_architecture_correctness as corr
import zero_roof_construction as construction
import zero_balcony_guard_variants as variants
import zero_alpha_cutout as cut
import zero_loggia_repair as repair
import zero_integrated_cutouts as integrated

OUT=Path("artifacts/zero-external-balconies")
PREV=Path("artifacts/zero-integrated-cutouts")
ECON=Path("artifacts/zero-balcony-guard-variants/economy")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main(decoder):
    OUT.mkdir(parents=True,exist_ok=True)
    cases=arch.fixtures()
    styles={c["building_id"]:c["id"] for c in cases}
    op=tile.plan_facades
    try:
        tile.plan_facades=arch.styled_plan
        groups,t0,cells,_=tile.compile_scene(cases,audit_samples=2048)
    finally:
        tile.plan_facades=op
    table=list(t0)
    assert cells.tobytes()==(PREV/"facade_cells.bin").read_bytes()
    # Do NOT call any recess planner or mutate the canonical host shell.
    shell={role:{k:v.copy() for k,v in g.items()} for role,g in groups.items()}
    oldparts,oldmanifest,*_=cut.source_geometry()
    roofpaths=defaultdict(list)
    for p,r in zip(oldparts["roof"],[r for r in oldmanifest if r["kind"]=="roof"]):
        roofpaths[r["id"].split(":cutout_guard:")[0]].append(p)
    parts=defaultdict(list);placements=[];anchors=set();guards=[];removed=Counter()
    for c in cases:
        plan,fac,fail=arch.plan_modules(c);rm,rf=roof.plan(c)
        assert not fail and not rf
        af={a.id:a for a in plan.facades}
        mods=variants.variants(c,plan,construction.improve(c,fac,rm),True)
        expected={o.id for a in plan.facades for o in a.openings if o.role=="balcony_door"}
        found=set()
        for m in mods:
            if m["kind"]=="roof_guard":
                paths=roofpaths[c["building_id"]];assert paths
                parts["roof"].extend(paths);removed["roof"]+=m["triangles"]
                continue
            if m["kind"].startswith("balcony_"):
                a=af[m["facade"]];n=np.asarray(a.outward);origin=np.asarray(a.start)
                v=np.concatenate([q.reshape(-1,3) for _,_,q in m["geometry"]])
                depth=(v[:,:2]-origin)@n
                assert depth.min()>=-1e-8 and depth.max()>.5
                assert m["anchor"] not in anchors
                anchors.add(m["anchor"]);found.add(m["anchor"])
                # Thicken only the lower solid slab downwards, keep threshold
                # and enclosure contact unchanged. Upper cap is untouched.
                adjusted=[]
                for k,mat,t in m["geometry"]:
                    t=t.copy()
                    if k=="slab" and len(t)==12:
                        lo,hi=t[:,:,2].min(),t[:,:,2].max()
                        assert hi-lo>.05
                        t[:,:,2][np.abs(t[:,:,2]-lo)<1e-8]-=.035
                    adjusted.append((k,mat,t))
                m=dict(m,geometry=adjusted)
                for other in mods:
                    if other["id"]!=m["id"] and other["kind"] not in ("continuous_parapet","cornice","roof_guard","pitched_overhang","open_gutters"):
                        assert not corr.overlaps(m,other),(m["id"],other["id"])
                if m["kind"]=="balcony_open":
                    rails=[p for p in m["geometry"] if p[0]=="rail"]
                    assert len(rails)==3
                    edges=[];heights=[]
                    for _,_,t in rails:
                        vv=np.unique(t.reshape(-1,3),axis=0)
                        lo,hi=vv[:,2].min(),vv[:,2].max()
                        bottom=vv[np.abs(vv[:,2]-lo)<1e-7]
                        assert len(bottom)==2
                        edges.append(tuple(bottom+np.asarray(c["offset"])));heights.append(hi-lo)
                    assert np.ptp(heights)<1e-8
                    path,closed=cut.chain(edges);assert not closed
                    q=cut.strip(path,float(heights[0]),cut.PERIOD["balcony"])
                    assert len(q[2])==sum(len(t) for _,_,t in rails)
                    parts["balcony"].append(q);removed["balcony"]+=len(q[2])
                    guards.append(dict(anchor=m["anchor"],placement=m["id"],**q[3]))
                    m=dict(m,geometry=[p for p in m["geometry"] if p[0]!="rail"],cutout_guard=True)
                m.update(triangles=sum(len(t) for _,_,t in m["geometry"]),bounds=combined.bounds(m).tolist())
            combined.append_compact(groups,table,c,m)
            placements.append({k:v for k,v in m.items() if k!="geometry"})
        assert found==expected
    assert len(anchors)==86
    assert all("recess" not in m["kind"] for m in placements)
    assert table[:len(t0)]==t0
    for role,g in shell.items():
        for k,v in g.items():
            np.testing.assert_array_equal(groups[role][k][:len(v)],v)
    path=OUT/"tile_opaque.glb";tile.export_glb(groups,path)
    v,u,ids,_=tile.read_glb(path,groups,table)
    assert len(v)//3==sum(len(g["f"]) for g in groups.values())
    (OUT/"facade_cells.bin").write_bytes(cells.tobytes())
    samples=pack_table(table,cells,OUT/"facade_params.bin")
    check=json.loads(subprocess.run([decoder,str(OUT/"facade_params.bin"),str(OUT/"facade_cells.bin"),str(len(LAYERS))],capture_output=True,text=True,check=True).stdout)
    assert check["records"]==len(table) and check["cells"]==len(cells)
    (OUT/"parameters.json").write_text(json.dumps(table,indent=2))
    (OUT/"placements.json").write_text(json.dumps(placements,indent=2))
    (OUT/"cutout_placements.json").write_text(json.dumps(guards,indent=2))
    masks={k:cut.pattern(k,512) for k in parts}
    meshes={k:cut.join(p) for k,p in parts.items()}
    cutpath=OUT/"roof_balcony_cutout.glb";cut.export(cutpath,meshes,masks)
    actual=cut.read(cutpath,meshes,masks)
    added={k:len(q[2]) for k,q in actual.items()}
    assert removed["roof"]==912 and added["roof"]==20
    assert added["balcony"]==removed["balcony"]==6*len(guards)
    assert len(guards)==sum(m["kind"]=="balcony_open" for m in placements)
    baseline=json.loads((PREV/"report.json").read_text())
    scene=(v,u,ids,table)
    for k,(vv,uu,ff,im) in actual.items():
        scene=integrated.append_layer(scene,vv,uu,ff,k)
    assert len(scene[0])//3==len(v)//3+sum(added.values())
    # Actual reloaded GLB and alpha-mask mip depth check.
    lab=cut.Lab()
    objects=[(lab.mesh(v,np.zeros((len(v),2)),np.arange(len(v),dtype=np.uint32).reshape(-1,3)),None,(.65,.64,.59))]
    for k,(vv,uu,ff,im) in actual.items():
        levels,_=cut.mip_chain(im,True)
        d=OUT/"mips"/k;d.mkdir(parents=True,exist_ok=True)
        for i,mip in enumerate(levels):Image.fromarray(mip).save(d/f"{i}.png")
        objects.append((lab.mesh(vv,uu,ff),lab.texture(levels,8.),(0,0,0)))
    cam=((61,6,10),4.3,(.35,-1,.45))
    im,depth=lab.render(objects,*cam);_,other=lab.render(list(reversed(objects)),*cam)
    np.testing.assert_allclose(depth,other,rtol=0,atol=1e-6)
    Image.fromarray(im).save(OUT/"egl_whole_scene.png")
    joints={r["facade_id"]:corr.panel_joints(r,cells) for r in t0 if r["role"]=="wall" and styles[r["building"]]=="panel_block"}
    maps={k:texture_maps(MATERIALS[k],256)["basecolor"] for k in MATERIALS}
    original=ref.shader;base=variants.variant_shader(original,cells,styles,joints)
    alpha={k:repair.alpha_visibility(q[3]) for k,q in actual.items()}
    def kind(r):return r.get("accessory","").removeprefix("cutout_")
    def visible(s,r):
        k=kind(r)
        return alpha[k](s,dict(r,accessory="cutout_rail")) if k in alpha else np.ones(len(s),bool)
    def two(r):return kind(r) in alpha
    def shade(s,r,maps,mode,dense=False):
        k=kind(r)
        if k in alpha:
            col=np.broadcast_to(srgb_decode(np.array(cut.RGB[k])/255.),(len(s),3)).copy()
            return col,np.zeros_like(col)
        col,em=base(s,r,maps,mode,dense)
        if r.get("accessory")=="glazed":
            x,y=s.T/np.asarray(r["span"])[:,None]
            glass=y>.32
            bars=max(2,round(r["span"][0]/.65))
            frame=glass&((np.mod(x*bars,1)<.06)|(x>.97)|(y>.975)|(np.abs(y-.33)<.018))
            col[glass]=srgb_decode(np.array([.22,.33,.37]))
            col[frame]=srgb_decode(np.array([.78,.76,.72]))
        return col,em
    cams=[("whole", (61,6,10),4.3,(.35,-1,.45)),
          ("panel_full",(58,6,10.5),21.,(.35,-1,.22)),
          ("brick_balconies",(107,0,9),32.,(.25,-1,.16)),
          ("roof",(58,5,21),28.,(.25,-1,.42)),
          ("aerial",(61,6,10),3.0,(.3,-1,1.8))]
    sheet=Image.new("RGB",(1280,640*len(cams)));views=[]
    ref.shader=shade
    try:
        for i,(name,center,ppm,view) in enumerate(cams):
            lighting,_=repair.shadow_lighting(scene,maps,center,ppm,(.5,-.7,.63),visible,two)
            im,_=ref.render(*scene,maps,"day",center_override=center,ppm_override=ppm,view_override=view,
                visibility=visible,two_sided=two,lighting=lighting)
            im.save(OUT/f"{name}_external.png")
            prev=Image.open(PREV/f"{name}_integrated.png").convert("RGB")
            views.append(dict(view=name,changed_rgb_pixels=int(np.any(np.asarray(prev)!=np.asarray(im),axis=2).sum())))
            for j,(label,img) in enumerate((("stage19",prev),("external only",im))):
                ImageDraw.Draw(img).text((8,8),label,fill=(0,0,0));sheet.paste(img,(j*640,i*640))
        for r in t0:
            if r["role"]!="wall":continue
            s=np.random.default_rng(r["seed"]).random((1024,2))*[r["length"],r["height"]]
            for a,b in zip(base(s,r,maps,"night"),shade(s,r,maps,"night")):
                np.testing.assert_array_equal(a,b)
        im,_=ref.render(*scene,maps,"night",center_override=cam[0],ppm_override=cam[1],view_override=cam[2],visibility=visible,two_sided=two)
        im.save(OUT/"night_external.png")
    finally:ref.shader=original
    sheet.save(OUT/"comparison.png")
    report=dict(schema="pcg.research.external-balconies.v1",previous_triangles=baseline["candidate_triangles"],
        candidate_triangles=len(scene[0])//3,external_anchors=len(anchors),recessed_anchors=0,
        open_cutout_guards=len(guards),removed_opaque=dict(removed),added_cutout=added,
        opening_triangles_added=0,cpp_abi=True,abi_samples=samples,
        host_shell_prefix_preserved=True,cells_sha256=sha(OUT/"facade_cells.bin"),
        egl_draw_order_max_depth_delta=float(np.max(np.abs(depth-other))),views=views,
        limitations=["Three synthetic rectangular fixtures; CPU diffuse and software EGL only.",
          "Historic stages retain recesses for comparison only; this output has no recesses.",
          "Door plus adjacent window motif still pending; canonical cell bytes unchanged.",
          "Full/partial glazing remains opaque approximation; no native UNIGINE acceptance."])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    files=[path,cutpath,OUT/"facade_cells.bin",OUT/"facade_params.bin",OUT/"parameters.json",OUT/"placements.json",OUT/"cutout_placements.json"]+sorted((OUT/"mips").rglob("*.png"))
    (OUT/"bundle.json").write_text(json.dumps(dict(schema=report["schema"],hashes={str(p.relative_to(OUT)):sha(p) for p in files}),indent=2))
    print("EXTERNAL_BALCONY_REPORT "+json.dumps(report),flush=True)
    thumb=sheet.copy();thumb.thumbnail((1200,1600));buf=io.BytesIO();thumb.save(buf,format="JPEG",quality=78,optimize=True)
    data=base64.b64encode(buf.getvalue()).decode()
    for i in range(0,len(data),4000):print(f"EXTERNAL_BALCONY_PREVIEW_{i//4000:03d} "+data[i:i+4000],flush=True)

if __name__=="__main__":main(sys.argv[1])
