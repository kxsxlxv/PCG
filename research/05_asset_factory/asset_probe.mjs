// Research 05: independent procedural asset probe, no dependencies except Node.js for CLI.
function factory() {
  // Coordinates: glTF RH Y-up, +Z outward, origin at mounting-plane bottom center.
  const mats=[
    ["ivory_frame",[.88,.86,.80,1],0,.38],["glass_proxy",[.14,.24,.31,1],.15,.19],
    ["dark_metal",[.20,.23,.24,1],.65,.42],["concrete",[.57,.54,.49,1],0,.87],
    ["painted_metal",[.79,.80,.76,1],.35,.52],["brick",[.43,.19,.12,1],0,.85],
    ["rubber",[.045,.05,.05,1],0,.85],["wood",[.37,.22,.12,1],0,.65],
    ["zinc",[.56,.60,.62,1],.75,.36],["shadow",[.08,.09,.09,1],0,.9]
  ];
  const sub=(a,b)=>a.map((x,i)=>x-b[i]), dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),
    cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],
    norm=a=>Math.hypot(...a), unit=a=>a.map(x=>x/norm(a));
  const kinds=["casement_window","sliding_door","open_balcony","glazed_balcony","ac_unit","chimney","roof_vent","entrance_door"];
  function generate(kind, variant=0, lod=0) {
    if(!kinds.includes(kind)||!Number.isInteger(variant)||variant<0||variant>4||![0,1,2].includes(lod))throw Error("invalid recipe");
    const g={kind,variant,lod,groups:{},mount:kind==="chimney"||kind==="roof_vent"?"roof":"wall"};
    function face(m, p) {
      const a=unit(sub(p[1],p[0])), n=unit(cross(sub(p[1],p[0]),sub(p[2],p[0]))), b=cross(n,a);
      const o=g.groups[m]??=( {p:[],n:[],uv:[],ix:[]} );
      const k=o.p.length/3;
      for(const v of p){o.p.push(...v);o.n.push(...n);o.uv.push(dot(sub(v,p[0]),a),dot(sub(v,p[0]),b));}
      for(let i=1;i<p.length-1;i++)o.ix.push(k,k+i,k+i+1);
    }
    function box(m,x,y,z,w,h,d){
      if(Math.min(w,h,d)<=0)throw Error("invalid box");
      const a=x-w/2,b=x+w/2,c=y,e=y+h,f=z-d/2,k=z+d/2;
      const p=[[a,c,f],[b,c,f],[b,e,f],[a,e,f],[a,c,k],[b,c,k],[b,e,k],[a,e,k]];
      for(const ids of [[3,2,1,0],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]])face(m,ids.map(i=>p[i]));
    }
    function cyl(m,x,y,z,r,h,n,axis="y"){
      const at=(i,t)=>{const a=2*Math.PI*i/n;return axis==="y"?[x+r*Math.cos(a),y+t,z+r*Math.sin(a)]:[x+r*Math.cos(a),y+r*Math.sin(a),z+t];};
      const lo=Array.from({length:n},(_,i)=>at(i,0)), hi=Array.from({length:n},(_,i)=>at(i,h));
      // angle around Y produces -Y cap; around Z produces +Z cap.
      if(axis==="y"){face(m,lo);face(m,hi.slice().reverse());for(let i=0;i<n;i++)face(m,[lo[i],hi[i],hi[(i+1)%n],lo[(i+1)%n]]);}
      else{face(m,lo.slice().reverse());face(m,hi);for(let i=0;i<n;i++)face(m,[lo[i],lo[(i+1)%n],hi[(i+1)%n],hi[i]]);}
    }
    const q=variant-2;
    function frame(w,h,y,z,m=0,div=1){
      const t=.065,d=.09;
      box(m,-w/2+t/2,y,z,t,h,d);box(m,w/2-t/2,y,z,t,h,d);
      box(m,0,y,z,w-2*t,t,d);box(m,0,y+h-t,z,w-2*t,t,d);
      const span=(w-2*t)/div;
      for(let i=0;i<div;i++){
        const x=-w/2+t+(i+.5)*span;
        box(1,x,y+t,z-.025,span-t*.45,h-2*t,.02);
        if(lod===0){for(const s of [-1,1])box(6,x+s*(span-t*.45)/2,y+t,z+.017,.012,h-2*t,.018);}
        if(i>0)box(m,x-span/2,y+t,z,.05,h-2*t,d);
      }
      if(lod===0)box(2,w*.19,y+h*.43,z+.06,.035,.15,.045);
    }
    if(kind==="casement_window"||kind==="sliding_door"){
      const w=(kind==="casement_window"?1.5:2.4)+q*.15,h=kind==="casement_window"?1.35+q*.05:2.15;
      if(lod===2)box(1,0,0,.035,w,h,.07);
      else{frame(w,h,0,.035,0,kind==="casement_window"?2:3);if(kind==="casement_window")box(3,0,0,.065,w+.12,.045,.20);else box(8,0,0,.04,w,.035,.12);}
    }else if(kind==="open_balcony"||kind==="glazed_balcony"){
      const w=2.7+q*.25,d=1.1+q*.06,h=1.05,t=.16;
      box(3,0,0,d/2,w,t,d);
      if(kind==="glazed_balcony"){
        if(lod===2)box(1,0,t,d-.035,w,2.15,.07);
        else{frame(w,2.15,t,d-.055,0,3);for(const s of [-1,1])box(1,s*(w/2-.035),t,d/2,.07,2.15,d-.1);box(4,0,t+2.15,d/2,w,.08,d);}
      }else if(lod===2)box(2,0,t,d-.035,w,h,.07);
      else{
        box(2,0,t+h-.045,d-.04,w,.045,.07);
        for(const s of [-1,1]){box(2,s*(w/2-.025),t+h-.045,d/2,.05,.045,d);box(2,s*(w/2-.025),t,d-.04,.055,h,.055);}
        const count=lod===0?Math.ceil(w/.14):5;
        for(let i=1;i<count;i++)box(2,-w/2+w*i/count,t,d-.04,.025,h-.045,.025);
        for(const s of [-1,1])for(let i=1;i<(lod===0?7:3);i++)box(2,s*(w/2-.025),t,d*i/(lod===0?7:3),.025,h-.045,.025);
      }
    }else if(kind==="ac_unit"){
      const w=.82+q*.06,h=.56,d=.31;
      box(4,0,.09,d/2+.04,w,h,d);
      if(lod<2){box(9,-w*.14,.16,d+.047,w*.55,.43,.014);cyl(2,-w*.14,.375,d+.055,.17,.018,lod===0?24:12,"z");
        if(lod===0){for(let i=0;i<9;i++)box(4,-w*.14,.22+i*.038,d+.076,.32,.013,.018);for(let i=0;i<5;i++)box(2,w*.34,.18+i*.07,d+.04,.10,.018,.01);}
        for(const s of [-1,1])box(2,s*w*.3,0,d/2,.055,.09,d+.08);
      }
    }else if(kind==="chimney"){
      const w=.55+q*.045,h=1.05+q*.09;
      box(5,0,0,0,w,h,w);box(8,0,h,0,w+.13,.09,w+.13);
      if(lod<2){box(9,0,h+.09,0,w*.45,.015,w*.45);if(lod===0){for(const s of [-1,1])box(8,s*w*.38,h+.09,0,.045,.16,.045);box(8,0,h+.25,0,w+.1,.045,w+.1);}}
    }else if(kind==="roof_vent"){
      const r=.16+q*.015,h=.65+q*.045,n=lod===0?20:lod===1?10:6;
      box(8,0,0,0,r*3,.045,r*3);cyl(8,0,.04,0,r,h-.04,n);cyl(2,0,h,0,r*1.5,.07,n);
      if(lod===0){for(let i=0;i<8;i++){const a=i*Math.PI/4;box(2,r*Math.cos(a),h-.09,r*Math.sin(a),.025,.10,.025);}}
    }else if(kind==="entrance_door"){
      const w=1.15+q*.08,h=2.15;
      if(lod===2)box(7,0,0,.045,w,h,.09);
      else{box(7,0,.045,.035,w-.13,h-.11,.07);for(const s of [-1,1])box(0,s*(w/2-.0325),0,.045,.065,h,.10);box(0,0,h-.065,.045,w-.13,.065,.10);
        box(1,0,h*.52,.078,w*.48,h*.31,.018);
        if(lod===0){box(2,w*.32,.97,.105,.14,.035,.055);box(2,-w*.40,.19,.10,.04,.16,.04);box(2,-w*.40,1.78,.10,.04,.16,.04);}
      }
    }
    return g;
  }
  function validate(g){
    let triangles=0,vertices=0,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for(const o of Object.values(g.groups)){
      const n=o.p.length/3;
      if(n!==o.n.length/3||n!==o.uv.length/2||o.ix.length%3!==0||![...o.p,...o.n,...o.uv].every(Number.isFinite))throw Error("attribute mismatch");
      vertices+=n;triangles+=o.ix.length/3;
      for(let i=0;i<n;i++)for(let j=0;j<3;j++){min[j]=Math.min(min[j],o.p[3*i+j]);max[j]=Math.max(max[j],o.p[3*i+j]);}
      for(let i=0;i<o.ix.length;i+=3){
        const ids=o.ix.slice(i,i+3);
        if(ids.some(k=>!Number.isInteger(k)||k<0||k>=n))throw Error("bad index");
        const [a,b,c]=ids.map(k=>o.p.slice(k*3,k*3+3));
        const nn=cross(sub(b,a),sub(c,a)), vn=o.n.slice(ids[0]*3,ids[0]*3+3);
        if(norm(nn)<1e-9||dot(nn,vn)<=0||Math.abs(norm(vn)-1)>1e-8)throw Error("degenerate or winding");
        const uv=ids.map(k=>o.uv.slice(k*2,k*2+2));
        if(Math.abs((uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[2][0]-uv[0][0])*(uv[1][1]-uv[0][1]))<1e-9)throw Error("degenerate UV");
      }
    }
    return {triangles,vertices,material_slots:Object.keys(g.groups).length,min,max};
  }
  function base64(bytes){
    const alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let s="";for(let i=0;i<bytes.length;i+=3){const a=bytes[i],b=bytes[i+1]??0,c=bytes[i+2]??0,v=(a<<16)|(b<<8)|c;s+=alphabet[(v>>18)&63]+alphabet[(v>>12)&63]+(i+1<bytes.length?alphabet[(v>>6)&63]:"=")+(i+2<bytes.length?alphabet[v&63]:"=");}return s;
  }
  function gltf(assets, gallery=true){
    const j={asset:{version:"2.0",generator:"PCG Asset Factory probe v1"},scene:0,scenes:[{nodes:[]}],nodes:[],meshes:[],materials:mats.map(([name,baseColorFactor,metallicFactor,roughnessFactor])=>({name,pbrMetallicRoughness:{baseColorFactor,metallicFactor,roughnessFactor}})),accessors:[],bufferViews:[],buffers:[]};
    const bytes=[];
    function add(values,components,integer=false,position=false){
      const start=bytes.length, a=new DataView(new ArrayBuffer(values.length*4));
      for(let i=0;i<values.length;i++)integer?a.setUint32(i*4,values[i],true):a.setFloat32(i*4,values[i],true);
      for(const b of new Uint8Array(a.buffer))bytes.push(b);
      const vi=j.bufferViews.push({buffer:0,byteOffset:start,byteLength:a.byteLength,target:integer?34963:34962})-1;
      const ac={bufferView:vi,componentType:integer?5125:5126,count:values.length/components,type:components===1?"SCALAR":"VEC"+components};
      if(position){ac.min=[Infinity,Infinity,Infinity];ac.max=[-Infinity,-Infinity,-Infinity];for(let i=0;i<values.length;i++){const k=i%3,v=Math.fround(values[i]);ac.min[k]=Math.min(ac.min[k],v);ac.max[k]=Math.max(ac.max[k],v);}}
      return j.accessors.push(ac)-1;
    }
    for(const [i,g] of assets.entries()){
      const name=`${g.kind}_v${g.variant}_lod${g.lod}`, primitives=[];
      for(const [m,o] of Object.entries(g.groups))primitives.push({attributes:{POSITION:add(o.p,3,false,true),NORMAL:add(o.n,3),TEXCOORD_0:add(o.uv,2)},indices:add(o.ix,1,true),material:+m,mode:4});
      const mesh=j.meshes.push({name,primitives})-1;
      const node={name,mesh,extras:{kind:g.kind,variant:g.variant,lod:g.lod,mount:g.mount}};
      if(gallery)node.translation=[(i%4)*4,0,Math.floor(i/4)*4];
      j.scenes[0].nodes.push(j.nodes.push(node)-1);
    }
    j.buffers.push({byteLength:bytes.length,uri:"data:application/octet-stream;base64,"+base64(bytes)});
    return j;
  }
  function run(){
    const tests=[], galleries=[];
    for(const kind of kinds)for(let v=0;v<5;v++){
      const stats=[];
      for(let l=0;l<3;l++){const g=generate(kind,v,l),s=validate(g);if(JSON.stringify(g)!==JSON.stringify(generate(kind,v,l)))throw Error("nondeterministic");stats.push(s);}
      if(stats[1].triangles>stats[0].triangles||stats[2].triangles>stats[1].triangles)throw Error("non-monotonic LOD "+kind);
      tests.push({kind,variant:v,lods:stats});
    }
    for(let l=0;l<3;l++)galleries.push(gltf(kinds.map(k=>generate(k,2,l))));
    const totals=[0,1,2].map(l=>tests.filter(t=>t.variant===2).reduce((s,t)=>s+t.lods[l].triangles,0));
    return {tests,galleries,totals};
  }
  return {generate,validate,gltf,run,kinds};
}
export { factory };
import {mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href){
 const out=process.argv[2]||'generated';mkdirSync(out,{recursive:true});
 const f=factory(),r=f.run();
 for(let l=0;l<3;l++)writeFileSync(join(out,'gallery_lod'+l+'.gltf'),JSON.stringify(r.galleries[l]));
 for(const kind of f.kinds)for(let l=0;l<3;l++)writeFileSync(join(out,kind+'_lod'+l+'.gltf'),JSON.stringify(f.gltf([f.generate(kind,2,l)],false)));
 writeFileSync(join(out,'test_results.json'),JSON.stringify({tests:r.tests,default_triangles:r.totals},null,2));
 console.log('Validated '+r.tests.length*3+' meshes; gallery triangles '+r.totals.join(', '));
}
