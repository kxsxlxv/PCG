"""Diagnostic views of exported and reloaded rectangular shells."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pcg.contracts import Rectangle
from pcg.geometry import build_shell

def main():
    out=Path("artifacts/core-shell")
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    for roof,rise in [("flat",0),("gable",2)]:
        b=Rectangle(roof,12,8,6,roof,rise)
        s=build_shell(b)
        mesh=trimesh.Trimesh(vertices=s.vertices,faces=[t for _,t in s.triangles()],process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent
        roles={x.id:x.role for x in s.surfaces}
        palette={"base":[110,110,110,255],"wall":[218,205,180,255],"roof":[150,75,55,255]}
        colors=np.array([palette[roles[k]] for k,_ in s.triangles()],dtype=np.uint8)
        mesh.unmerge_vertices()
        mesh.visual.vertex_colors=np.repeat(colors,3,axis=0)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
        path=out/(roof+".glb")
        path.write_bytes(mesh.export(file_type="glb"))
        loaded=trimesh.load(path,force="mesh",process=False)
        loaded.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        np.testing.assert_allclose(loaded.bounds,[[0,0,0],[12,8,6+rise]],atol=1e-6)
        (out/(roof+".json")).write_text(json.dumps({"schema_version":1,"compiler_version":"0.1.0","input_digest":b.digest(),"shell":asdict(s)},sort_keys=True,indent=2))
        sheet=Image.new("RGB",(960,960),"white")
        for i,(name,view,up) in enumerate([("front",[0,-1,0],[0,0,1]),("side",[1,0,0],[0,0,1]),("top",[0,0,1],[0,1,0]),("iso",[1,-1,0.8],[0,0,1])]):
            view=np.array(view,dtype=float);view/=np.linalg.norm(view)
            right=np.cross(up,view);right/=np.linalg.norm(right)
            up=np.cross(view,right)
            v=loaded.vertices-np.array([6,4,4])
            xy=np.column_stack((v@right,-v@up))*24+240
            depth=v@view
            canvas=Image.new("RGB",(480,480),(245,243,238));draw=ImageDraw.Draw(canvas)
            # Convex shells only: backface culling makes visible face interiors disjoint.
            count=0
            for k in np.argsort(depth[loaded.faces].mean(axis=1)):
                if loaded.face_normals[k]@view<=1e-8:continue
                face=loaded.faces[k]
                c=loaded.visual.vertex_colors[face,:3].mean(axis=0)
                draw.polygon([tuple(p) for p in xy[face]],fill=tuple(map(int,c)),outline=(80,80,80))
                count+=1
            assert count>0
            draw.text((12,12),roof+" / "+name,fill=(0,0,0))
            canvas.save(out/(roof+"_"+name+".png"))
            sheet.paste(canvas,((i%2)*480,(i//2)*480))
        sheet.save(out/(roof+"_contact.png"))
        report.append({"roof":roof,"volume_m3":12*8*(6+rise/2),"triangles":len(s.triangles()),"surfaces":len(s.surfaces),"edges":len(s.edges)})
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=="__main__":
    main()
    from render_l2_massing import main as render_l2
    render_l2()
