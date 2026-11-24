// CPU-only synthetic streaming and projected-error model; Node.js CLI.
function experiment(){
 const side=40, size=512, N=side*side, count=64*N, dt=.1, duration=60, warm=10, radius=1500, fineBudget=96, coarseMiB=N*.09375;
 const tiles=Array.from({length:N},(_,id)=>({id,x:id%side*size,y:Math.floor(id/side)*size,mb:1+((id*2654435761>>>0)%5)*.125}));
 const distance=(tile,x,y)=>Math.hypot(Math.max(tile.x-x,x-tile.x-size,0),Math.max(tile.y-y,y-tile.y-size,0));
 function position(t,route){t=Math.max(0,t);return route==="straight"?[2000+250*t,10000,250,0]:[2000+250*t,10000+1800*Math.sin(t/7),250,1800/7*Math.cos(t/7)];}
 function query(x,y){return tiles.filter(q=>distance(q,x,y)<=radius).map(q=>q.id);}
 function sim(route,bw,lookahead){
   const cache=new Map(),everUsed=new Set();let job=null,used=0,peak=0,miss=0,demand=0,zeroTicks=0,loads=0,readMiB=0,evictions=0,wasted=0,worst=0,maxDemand=0,late=0,transitions=0,prior=new Set();
   for(let step=-Math.round(warm/dt);step<Math.round(duration/dt);step++){
     const t=step*dt,[x,y,vx,vy]=position(t,route),active=new Set(query(x,y)),wanted=new Map();
     for(const id of active)wanted.set(id,0);
     if(t>=0)for(let l=1;l<=lookahead;l++)for(const id of query(x+l*vx,y+l*vy))if(!wanted.has(id))wanted.set(id,l);
     if(job&&job.finish<=t+1e-8){cache.set(job.id,{last:t,used:false});used+=tiles[job.id].mb;loads++;job=null;}
     for(const id of active)if(cache.has(id)){const item=cache.get(id);item.last=t;item.used=true;everUsed.add(id);}
     if(t>=0){
       let missing=0;for(const id of active)if(!cache.has(id))missing++;
       demand+=active.size;miss+=missing;zeroTicks+=missing===0?1:0;worst=Math.max(worst,missing);maxDemand=Math.max(maxDemand,active.size);
       for(const id of active)if(!prior.has(id)){transitions++;late+=!cache.has(id)?1:0;}
     }
     prior=active;
     if(!job){
       const candidates=[...wanted.keys()].filter(id=>!cache.has(id)).sort((a,b)=>wanted.get(a)-wanted.get(b)||distance(tiles[a],x,y)-distance(tiles[b],x,y)||a-b);
       if(candidates.length){
         const id=candidates[0],mb=tiles[id].mb;
         const victims=[...cache.keys()].filter(k=>!active.has(k)).sort((a,b)=>(wanted.has(a)?1:0)-(wanted.has(b)?1:0)||cache.get(a).last-cache.get(b).last||a-b);
         while(used+mb>fineBudget+1e-9&&victims.length){const v=victims.shift();if(!cache.get(v).used)wasted++;cache.delete(v);used-=tiles[v].mb;evictions++;}
         if(used+mb<=fineBudget+1e-9){job={id,finish:t+.075+mb/bw};readMiB+=mb;}
       }
     }
     peak=Math.max(peak,used+(job?tiles[job.id].mb:0));if(peak>fineBudget+1e-9)throw Error("budget exceeded");
   }
   for(const item of cache.values())if(!item.used)wasted++;
   return {route,bandwidth_mib_s:bw,lookahead_s:lookahead,demand_tile_seconds:demand*dt,missing_fine_tile_seconds:miss*dt,detail_coverage_pct:100*(1-miss/demand),fully_detailed_ticks_pct:100*zeroTicks/(duration/dt),worst_missing_tiles:worst,max_demand_tiles:maxDemand,entry_count:transitions,late_entries:late,loads_completed:loads,read_mib_including_warmup_and_pending:readMiB,evictions,completed_loads_never_demanded:wasted,peak_fine_payload_mib:peak,coarse_payload_mib:coarseMiB,holes:0};
 }
 const tests=[];
 for(const route of ["straight","curved"])for(const bw of [6,12,24])for(const look of [0,3])tests.push(sim(route,bw,look));
 // Pure projected-error geometry estimate for 102400 uniformly spaced buildings, not frustum-visible triangles or measured GPU.
 const focal=1440/(2*Math.tan(Math.PI/6)), errors=[0,.12,.65,2.5], tris=[1800,360,80,24];
 const lods=[];
 for(const altitude of [50,300,1500,5000]){
   const bins=[0,0,0,0],cam=[10000,10000,altitude], buildingHeight=12;
   for(const q of tiles)for(let i=0;i<64;i++){
     const x=q.x+(i%8+.5)*64,y=q.y+(Math.floor(i/8)+.5)*64;
     if(Math.hypot(x-cam[0],y-cam[1])>8000)continue;
     const d=Math.max(1,Math.hypot(x-cam[0],y-cam[1],altitude-buildingHeight/2)-10);
     let l=0;for(let j=3;j>=1;j--)if(errors[j]*focal/d<=1.5){l=j;break;}
     bins[l]++;
   }
   const n=bins.reduce((a,b)=>a+b,0);lods.push({altitude_m:altitude,buildings:n,lod_counts:bins,triangles:bins.reduce((s,b,i)=>s+b*tris[i],0),all_lod0_triangles:n*tris[0]});
 }
 // Threshold jitter test: same metric sampled 600 times around threshold. 20% hysteresis.
 let a=false,b=false,na=0,nb=0;for(let i=0;i<600;i++){const e=1.5+.15*Math.sin(i*.23);const aa=e>1.5,bb=b?e>=1.2:e>1.8;na+=aa!==a;nb+=bb!==b;a=aa;b=bb;}
 return {assumptions:{buildings:count,tile_size_m:size,tile_count:N,buildings_per_tile:64,world_extent_m:side*size,demand_radius_m:radius,step_s:dt,flight_seconds:duration,warmup_s:warm,fine_cache_mib:fineBudget,coarse_mib:coarseMiB,fine_tile_mib:[1,1.5],service_latency_s:.075,flight_x_velocity_m_s:250,coarse_always_resident:true,simulation:"serial synthetic transfer with tick-quantized completion, demand-first, constant-velocity prediction; no real I/O, GPU, frustum or native nodes",lod_model:{vertical_fov_deg:60,viewport_height_px:1440,threshold_px:1.5,geometric_error_m:errors,triangles_per_building:tris,horizontal_radius_m:8000,distance:"bounding-sphere approximation",visibility:"all candidates in circle; no frustum, occlusion or pass multipliers"}},tests,lods,jitter:{samples:600,no_hysteresis_switches:na,hysteresis_switches:nb,threshold:1.5,refine:1.8,coarsen:1.2},limitations:["synthetic uniform city and payloads","coarse entire region assumed resident, no global HLOD","no native rendering, actual disk I/O, decoding, upload, activation or frame timing","preload overhead includes warmup and pending request","coarse holes zero by construction, not measured visual correctness","triangle counts estimates based on assumed error and per-building budgets"]};
}
import {writeFileSync} from 'node:fs';
const result=experiment();writeFileSync(process.argv[2]||'streaming_results.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
