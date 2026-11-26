"""Stage 21A: high-resolution same-mesh window and projecting enclosure study.
No changes to canonical apertures, geometry, material ABI or stage 20 output.
Paired balcony door/window envelopes are deliberately a subsequent study.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from pcg.far_emission import srgb_decode
import render_zero_geometry_facades as ref
import compile_zero_geometry_tile as tile
import zero_facade_architecture as arch
import zero_loggia_repair as repair

OUT=Path("artifacts/zero-window-visual")
FRAMES=np.array([[.80,.78,.73],[.84,.85,.82],[.68,.64,.56],[.46,.32,.23]])

def is_host_wall(record):
    # Integrated alpha-cutout records intentionally omit role, seed and span.
    # Accessories are not canonical facade records.
    return "accessory" not in record and record.get("role")=="wall"

def candidate(base,cells):
    def shade(s,r,maps,mode,dense=False):
        col,em=base(s,r,maps,mode,dense)
        if not len(s):return col,em
        if r.get("accessory") in ("glazed","partial_upper"):
            w,h=r["span"]
            x,y=s.T/np.array([w,h])[:,None]
            key=ref.seed(r["placement"])
            lower=([.26,.32,.38][key%3] if r["accessory"]=="glazed" else 0.)
            # Shared placement appearance across faces; frame widths in metres.
            upper=y>=lower
            glass_height=max(.01,h*(1-lower))
            t=(y-lower)/max(.01,1-lower)
            n=max(1,round(w/[.55,.72,.90][(key//3)%3]))
            edge=np.minimum(x,1-x)*w
            mull=np.abs(x*n-np.rint(x*n))*w/n
            frame=upper&((edge<.035)|(mull<.025)|(t*glass_height<.045)|((1-t)*glass_height<.045))
            if key%4==0:frame|=upper&(np.abs(t-.68)*glass_height<.025)
            glass=upper&~frame
            pane=np.clip(np.floor(x*n),0,n-1)
            variation=.88+.16*((pane+(key%13))%5)/4
            rgb=np.column_stack((.16+.13*t,.23+.16*t,.27+.19*t))*variation[:,None]
            col[glass]=srgb_decode(np.clip(rgb[glass],0,1))
            col[frame]=srgb_decode(FRAMES[(key//7)%len(FRAMES)])
            # When the lower panel gets taller, repaint the former glass strip.
            if r["accessory"]=="glazed":
                panel=~upper
                palette=np.array([[.34,.23,.18],[.47,.17,.13],[.58,.59,.56],[.27,.30,.32]])
                col[panel]=srgb_decode(palette[(key//11)%len(palette)])
                col[panel&(np.mod(x*w/.14,1)<.10)]*=.78
            return col,em
        if not is_host_wall(r) or mode=="bare":return col,em
        role,var,local=tile.lookup(s,r,cells)
        opening=(role==1)|(role==2)
        if not opening.any():return col,em
        u,z=s.T;x,y=local.T
        b=np.floor((u-r["first_center"])/r["spacing"]+.5).astype(np.int64)
        f=np.floor(z/r["floor_h"]).astype(np.int64)
        # Stable across rebuilds, independent of GPU parameter record index.
        key=(b*73856093+f*19349663+int(r["seed"]))%1009
        layout=key%5
        ww=r["window_width"];hh=np.where(role==2,r["door_height"],r["window_height"])
        edge=np.minimum(np.minimum(x,1-x)*ww,np.minimum(y,1-y)*hh)
        reveal=opening&(edge<.035)
        frame=opening&(edge<.09)
        win=role==1;door=role==2
        # One pane, equal/asymmetric two panes, three panes, transom.
        vertical=((layout==1)&(np.abs(x-.5)*ww<.028))|((layout==2)&(np.abs(x-.36)*ww<.028))|((layout==3)&(np.minimum(np.abs(x-.33),np.abs(x-.67))*ww<.028))|((layout==4)&(np.abs(x-.60)*ww<.028))
        frame|=win&(vertical|((layout==4)&(np.abs(y-.72)*hh<.026)))
        frame|=door&(np.abs(y-.30)*hh<.032)
        opaque=door&(y<.30)&~frame
        glass=opening&~frame&~opaque
        colors=srgb_decode(FRAMES[(key//11)%len(FRAMES)])
        col[frame]=colors[frame]
        col[reveal]*=.57
        col[opaque]=colors[opaque]*.70
        t=np.clip(y,0,1)
        gain=.85+.25*((key//5)%7)/6
        rgb=np.column_stack((.13+.14*t,.20+.17*t,.25+.20*t))*gain[:,None]
        col[glass]=srgb_decode(np.clip(rgb[glass],0,1))
        # Preserve room variant; apply radiance only to the new pane masks.
        em[opening]=0
        if mode=="night":
            for i,name in enumerate(tile.VARIANTS):em[glass&(var==i)]=arch.EMISSION[name]
        return col,em
    return shade

def compare(scene,maps,cells,base,visible,two):
    OUT.mkdir(parents=True,exist_ok=True)
    improved=candidate(base,cells)
    views=[
        ("panel_full",(58,6,10.5),21.,(.35,-1,.22),"day",(1920,1080)),
        ("brick_balconies",(107,0,9),32.,(.25,-1,.16),"day",(1920,1080)),
        ("panel_close",(48.2,.5,10.5),90.,(.5,-1,.12),"day",(1280,1280)),
        ("panel_close_night",(48.2,.5,10.5),90.,(.5,-1,.12),"night",(1280,1280)),
    ]
    sheet=Image.new("RGB",(1280,400*len(views)),(230,230,230))
    draw=ImageDraw.Draw(sheet);rows=[]
    original=ref.shader
    try:
        # Direct regression for every wall: outside canonical apertures
        # unchanged, deterministic shading, valid colors.
        checked=0
        cutouts_checked=0
        for r in scene[3]:
            # Exercise the actual sparse records appended by append_layer,
            # through both shaders, including nonempty day/night samples.
            if r.get("accessory","").startswith("cutout_"):
                assert not is_host_wall(r)
                s=np.array([[.15,.25],[.5,.75],[1.,1.]])
                for mode in ("day","night"):
                    for a,b in zip(base(s,r,maps,mode),improved(s,r,maps,mode)):
                        np.testing.assert_array_equal(a,b)
                cutouts_checked+=1
                continue
            if not is_host_wall(r):continue
            s=np.random.default_rng(r["seed"]).random((4096,2))*[r["length"],r["height"]]
            role,_,_=tile.lookup(s,r,cells);other=(role!=1)&(role!=2)
            for mode in ("day","night"):
                old=base(s,r,maps,mode);new=improved(s,r,maps,mode);again=improved(s,r,maps,mode)
                for a,b,c in zip(old,new,again):
                    np.testing.assert_array_equal(a[other],b[other])
                    np.testing.assert_array_equal(b,c)
                    assert np.isfinite(b).all() and (b>=0).all()
            checked+=len(s)
        assert checked>0 and cutouts_checked==2
        for i,(name,center,ppm,view,mode,resolution) in enumerate(views):
            ref.shader=base
            # Same stage 20 shadow map for both lanes. Shadow-map res is
            # unchanged; higher resolution applies to the camera raster only.
            light=None
            if mode=="day":
                light,_=repair.shadow_lighting(scene,maps,center,ppm,(.5,-.7,.63),visible,two)
            imgs=[];depth=None
            for j,(label,shader) in enumerate((("baseline",base),("candidate",improved))):
                ref.shader=shader
                im,d=ref.render(*scene,maps,mode,center_override=center,ppm_override=ppm,view_override=view,
                    visibility=visible,two_sided=two,lighting=light,resolution=resolution)
                assert im.size==resolution
                if depth is None:depth=d
                else:np.testing.assert_array_equal(depth,d)
                im.save(OUT/f"{name}_{label}.png")
                imgs.append(np.asarray(im))
                thumb=im.copy();thumb.thumbnail((640,370))
                sheet.paste(thumb,(j*640+(640-thumb.width)//2,i*400+25))
                draw.text((j*640+8,i*400+5),f"{name} | {label}",fill=(0,0,0))
            changed=int(np.any(imgs[0]!=imgs[1],axis=2).sum())
            assert changed>0,name
            rows.append(dict(view=name,resolution=list(resolution),changed_rgb_pixels=changed,depth_identical=True))
    finally:ref.shader=original
    sheet.save(OUT/"comparison.png")
    report=dict(schema="pcg.research.window-visual.v1",views=rows,wall_samples_checked=checked,
        cutout_records_checked=cutouts_checked,opening_triangles_added=0,scene_triangles=len(scene[0])//3,canonical_apertures_unchanged=True,
        limitations=["CPU color and emission only; no native UNIGINE or actual transparency.",
        "Paired balcony door and window envelope not implemented yet.",
        "Shadow-map resolution remains 640; no AA or mip filter validation in this pass.",
        "Candidate visual acceptance requires human/model review of actual CI images."])
    (OUT/"report.json").write_text(json.dumps(report,indent=2))
    print("WINDOW_VISUAL_REPORT "+json.dumps(report),flush=True)
