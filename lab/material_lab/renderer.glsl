// GPU material laboratory, Z up. Linear textures; output linear HDR.
in vec3 world; in vec2 uv; flat in int rid; flat in vec3 gn; flat in vec3 tangent;
layout(location=0) out vec4 color;
uniform sampler2D metadata, openings, envMap, shadowMap;
uniform sampler2DArray albedoMaps, normalMaps, ormMaps, roomMaps, cutoutMaps;
uniform mat4 shadowVP;
uniform vec3 eye, viewDir, sunDir, sunColor, irradiance;
uniform float pixelsPerMeter, night, perspective, exposure;
uniform int passShadow, roomSource, lobbyLayer;
const float PI=3.14159265359;
vec4 M(int i,int c){return texelFetch(metadata,ivec2(c,i),0);}
vec4 O(int i,int c){return texelFetch(openings,ivec2(c,i),0);}
float hash(float n){return fract(sin(n*12.9898)*43758.5453);}
vec3 env(vec3 d,float rough){d=normalize(d);vec2 q=vec2(atan(d.y,d.x)/(2.*PI)+.5,acos(clamp(d.z,-1.,1.))/PI);return textureLod(envMap,q,rough*7.).rgb;}
float sunVis(vec3 p,vec3 n){vec4 clip=shadowVP*vec4(p+n*.025,1);vec3 q=clip.xyz/clip.w*.5+.5;if(any(lessThan(q,vec3(0)))||any(greaterThan(q,vec3(1))))return 1.;float a=0.;vec2 du=1./vec2(textureSize(shadowMap,0));for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)a+=q.z-.0004<=texture(shadowMap,q.xy+vec2(x,y)*du).r?1.:0.;return a/9.;}
vec3 brdf(vec3 a,vec3 n,vec3 v,float r,float metal,float ao,vec3 p,float direct){
    n=normalize(n);v=normalize(v);vec3 l=normalize(sunDir),h=normalize(l+v);float nv=max(dot(n,v),.001),nl=max(dot(n,l),0.),nh=max(dot(n,h),0.),vh=max(dot(v,h),0.);
    float a2=pow(max(r,.07),4.),den=nh*nh*(a2-1.)+1.;float D=a2/(PI*den*den);float k=pow(r+1.,2.)/8.;float G=nv/(nv*(1.-k)+k)*nl/(nl*(1.-k)+k);
    vec3 f0=mix(vec3(.04),a,metal);vec3 F=f0+(1.-f0)*pow(1.-vh,5.);vec3 spec=D*G*F/max(4.*nl*nv,.001);vec3 diff=(1.-F)*(1.-metal)*a/PI;
    vec3 Fi=f0+(max(vec3(1.-r),f0)-f0)*pow(1.-nv,5.);vec3 reflected=env(reflect(-v,n),r);
    // Approximate split-sum IBL, diffuse environment average supplied by CPU.
    vec3 ambient=irradiance*a*(1.-metal)*(1.-Fi)*ao+reflected*Fi*(1.-.55*r)*ao;
    return ambient+(diff+spec)*sunColor*nl*direct;
}
vec3 textureNormal(int layer,vec2 q,vec3 n,vec3 t,vec3 b){vec3 z=texture(normalMaps,vec3(q,float(layer))).rgb*2.-1.;z.xy*=.65;return normalize(t*z.x+b*z.y+n*max(.1,z.z));}
vec3 matTint(int layer,vec2 q,vec3 p,vec3 n,vec3 t,vec3 b,vec3 v,float direct,vec3 tint){vec3 a=texture(albedoMaps,vec3(q,float(layer))).rgb*tint;vec3 orm=texture(ormMaps,vec3(q,float(layer))).rgb;return brdf(a,textureNormal(layer,q,n,t,b),v,orm.g,orm.b,orm.r,p,direct);}
vec3 mat(int layer,vec2 q,vec3 p,vec3 n,vec3 t,vec3 b,vec3 v,float direct,float tint){return matTint(layer,q,p,n,t,b,v,direct,vec3(tint));}
// Window-frame profiles: four sash patterns x four coherent palettes.
vec3 roomRay(vec2 q,vec2 ratio,vec2 size,float key,bool lobby){
    float dep=lobby?3.8:2.6;vec3 d=vec3(-ratio/size,1.);vec3 p=vec3(clamp(q,0.,1.),0.);vec4 hit=loggiaExit(p,d,vec3(0),vec3(1,1,dep));
    vec2 h=hit.xy;int face=int(hit.w);float z=hit.z/dep;vec3 c;
    if(lobby && lobbyLayer>=0){int tile=4;vec2 st=vec2(h.x,1.-h.y);if(face==0){tile=3;st=vec2(z,1.-h.y);}else if(face==1){tile=5;st=vec2(1.-z,1.-h.y);}else if(face==2){tile=7;st=vec2(h.x,1.-z);}else if(face==3){tile=1;st=vec2(h.x,z);}vec2 atlas=(vec2(tile%3,tile/3)+clamp(st,.006,.994))/3.;c=texture(roomMaps,vec3(atlas,float(lobbyLayer))).rgb;return c;}
    if(roomSource>0){float dist=dep*.66/.34;vec2 st=(h-.5)*(dist/(dist+hit.z))+.5;int idx=int(key)%8;c=texture(roomMaps,vec3(st.x,1.-st.y,float(idx))).rgb;}
    else { // physically separate, neutral virtual room surfaces, no furniture pasted against glass
        c=vec3(.39,.34,.27);if(face==2)c=vec3(.23,.17,.12);else if(face==3)c=vec3(.55,.53,.47);else if(face==0||face==1)c*=.72;
        if(face==2)c*=.86+.14*cos(h.x*90.);if(face==5){float band=smoothstep(.03,.07,abs(h.x-.5));c=mix(vec3(.23,.22,.20),c,band);}
        c*=.64+.36*hash(key+12.);
    }
    float curtain=hash(key+17.);if(!lobby&&curtain>.68&&abs(q.x-.5)>.33)c=mix(c,vec3(.53,.48,.42)*(.82+.18*cos(q.x*115.)),.93);
    return c;
}
vec3 glass(vec3 c,vec3 n,vec3 v,float rough,float lit,float outer){float nv=clamp(dot(n,v),0.,1.);float f=.04+.96*pow(1.-nv,5.);vec3 reflection=env(reflect(-v,n),rough);float fmin=mix(.12,.22,outer);f=max(f,fmin);return c*(1.-f)*(night>.5?mix(.13,2.1,lit):.72)+reflection*f*(night>.5?.40:1.);}
vec3 window(vec2 s,vec4 rect,int role,float key,vec3 p,vec3 t,vec3 b,vec3 n,vec3 v,float direct,float state,int profile){
    vec2 size=rect.zw-rect.xy,q=(s-rect.xy)/size;vec2 ratio=clamp(vec2(dot(v,t),dot(v,b))/max(dot(v,n),.13),-5.,5.);int selected=profile>=0?profile:int(key);int windowLayout=selected%4;bool door=role==2||role==3;
    if(role==6){float e=min(min(q.x,1.-q.x)*size.x,min(q.y,1.-q.y)*size.y);
        float panel=min(abs(q.x-.19),abs(q.x-.81));float grain=.83+.17*sin(q.x*155.+sin(q.y*8.));
        float trim=e<.065?.54:(panel<.025||abs(q.y-.45)<.018?.75:1.);
        return mat(15,vec2(q.x,1.-q.y),p,n,t,-b,v,direct,grain*trim);
    }
    int steps=pixelsPerMeter*max(size.x,size.y)>55.?20:pixelsPerMeter*max(size.x,size.y)>20.?8:0;
    vec4 hit=windowPOM(q,ratio,size,windowLayout,door,steps);int part=int(hit.w+.5);
    vec3 frame=vec3(.63,.65,.62);int palette=(selected/4)%4;if(palette==1)frame=vec3(.43,.34,.25);if(palette==2)frame=vec3(.76,.77,.73);if(palette==3)frame=vec3(.25,.18,.12);
    vec2 m=hit.xy*size;float grain=.94+.06*sin(m.y*970.+sin(m.x*54.));frame*=grain;
    if(role==4){return mat(11,vec2(q.x,1.-q.y),p,n,t,-b,v,direct,1.);}
    if(part==3){vec3 c=roomRay(hit.xy,ratio,size,key,role==3);
        // state=0 preserves legacy hashed occupancy. The unified lane
        // encodes canonical off/warm/neutral as 1/2/3 and lobby as 4.
        float lit=state>0.?float(state>1.5):float(role==3||hash(key+3.)<.37);
        if(night>.5&&state>1.5&&state<3.5)c*=state<2.5?vec3(1.15,.84,.48):vec3(.86,1.02,1.12);
        return glass(c,n,v,.13,lit,0.);
    }
    vec3 a=part==1?vec3(.36,.34,.30):part==4?vec3(.48,.52,.53):part==5?frame*.73:frame;
    // Continuous narrow seals and a brushed sill; POM cannot alter silhouettes.
    float e=min(min(m.x,size.x-m.x),min(m.y,size.y-m.y));if(part==2&&e>.083&&e<.094)a*=.35;
    float ao=part==1?.66:part==2?.91:1.;return brdf(a,n,v,part==4?.36:.58,part==4?.45:0.,ao,p,direct);
}
bool contains(vec2 p,vec4 r){return all(greaterThanEqual(p,r.xy))&&all(lessThan(p,r.zw));}
vec3 wall(int id,vec2 s,vec3 p,vec3 v,float direct){
    vec4 a=M(id,0),b=M(id,1),c=M(id,2);vec3 t=M(id,4).xyz,n=M(id,5).xyz,up=vec3(0,0,1);int start=int(b.x),bays=int(b.y),floors=int(b.z),style=int(b.w);int profile=style>=3?int(M(id,7).x):-1;
    int floorId=int(floor(s.y/c.z)),bay=int(floor((s.x-c.x)/c.y+.5));int slot=start+floorId*bays+bay;vec4 rect=vec4(0),other=vec4(0),data=vec4(0);
    if(floorId>=0&&floorId<floors&&bay>=0&&bay<bays){rect=O(slot,0);other=O(slot,1);data=O(slot,2);if(contains(s,rect))return window(s,rect,int(data.x),data.z,p,t,up,n,v,direct,data.w,profile);if(contains(s,other))return window(s,other,int(data.y),data.z,p,t,up,n,v,direct,data.w,profile);}
    int entry=start+bays*floors;vec4 en=O(entry,0);if(contains(s,en))return window(s,en,int(O(entry,2).x),O(entry,2).z,p,t,up,n,v,direct,O(entry,2).w,profile);
    int layer=int(a.y);float repeat=a.z;vec2 q=vec2(s.x,-s.y)/repeat;float tint=.92+.08*sin(s.x*.73+sin(s.y*.67))+.06*sin(s.y*.45+s.x*.24);tint*=1.-.15*exp(-max(s.y,0.)/.36);
    if(style==0||style==4){float seam=min(abs(fract((s.x-c.x)/c.y)-.5)*c.y,min(fract(s.y/c.z),1.-fract(s.y/c.z))*c.z); // seams drawn only outside actual openings
        if(seam<.012)tint*=.70;
    }
    if(style<3){
        if(s.y<.6){layer=4;repeat=1.;q=vec2(s.x,-s.y)/repeat;tint*=.80;}
        return mat(layer,q,p,n,t,-up,v,direct,tint);
    }
    vec4 dims=M(id,6);vec3 accent=M(id,3).xyz;float floorZ=mod(max(s.y,0.),c.z);
    // Material zoning is subordinate to actual apertures and paired blocks.
    // No extra opening geometry, no unique per-building texture.
    tint*=.86+.06*hash(a.w+7.);
    if(style==5){float seam=min(mod(s.y,.19),.19-mod(s.y,.19));tint*=seam<.012?.54:1.;tint*=.92+.08*sin(floor(s.y/.19)*1.71);}
    if(style==6&&mod(float(bay),4.)==3.)tint*=.67;
    if(style==7&&floorZ>dims.z-.08&&floorZ<dims.z+dims.w+.08)tint*=.26;
    if(style==8&&floorZ>.16&&floorZ<c.z-.20)tint*=.45;
    if(style==9&&min(floorZ,c.z-floorZ)<.045)tint*=1.24;
    if(s.y<(style==8?.22:.46)){layer=4;q=vec2(s.x,-s.y)/1.6;tint*=.76;accent=vec3(1.);}
    if(s.y>dims.x-.14&&s.y<dims.x)tint*=1.14;
    return matTint(layer,q,p,n,t,-up,v,direct,accent*tint);
}
void main(){
    vec4 a=M(rid,0);int kind=int(a.x),layer=int(a.y);vec3 n=normalize(gn),t=normalize(tangent),b=normalize(cross(n,t));vec3 v=perspective>.5?normalize(eye-world):normalize(viewDir);if(dot(n,v)<0.)n=-n;
    if(kind==3){vec4 c=texture(cutoutMaps,vec3(layer<2?vec2(fract(uv.x),uv.y):uv,float(layer)));if(c.a<.5)discard;if(passShadow==1){color=vec4(1);return;}color=vec4(brdf(c.rgb,n,v,.65,.1,1.,world,sunVis(world,n)),1.);return;}
    if(passShadow==1){color=vec4(1);return;}
    float direct=sunVis(world,n);vec3 c;
    if(kind==1)c=wall(rid,uv,world,v,direct);
    else if(kind==2){
        vec4 opt=M(rid,1);vec3 o=M(rid,2).xyz,tt=M(rid,3).xyz,nn=M(rid,4).xyz;vec3 lo=M(rid,5).xyz,hi=M(rid,6).xyz;vec2 span=M(rid,7).xy;
        vec3 p=vec3(dot(world-o,tt),dot(world-o,nn),world.z-o.z);vec3 d=-vec3(dot(v,tt),dot(v,nn),v.z);vec4 hit=loggiaExit(p,d,lo,hi);
        float low=opt.y;float count=max(1.,round(span.x/.60));float x=uv.x/span.x,y=uv.y/span.y;
        bool frame=min(x,1.-x)*span.x<.03||abs(x*count-round(x*count))*span.x/count<.024||abs(y-low)*span.y<.034||(1.-y)*span.y<.038;
        vec3 frameColor=vec3(.63,.64,.60);if(opt.z<.25)frameColor=vec3(.28,.18,.12);
        if(frame||y<low){vec3 panel=frame?frameColor:vec3(.29,.32,.28)*(.89+.11*smoothstep(.03,.10,fract(uv.x/.12)));c=brdf(panel,n,v,.63,.03,.9,world,direct);}
        else{
            vec3 hp=o+tt*hit.x+nn*hit.y+vec3(0,0,hit.z);int face=int(hit.w+.5);vec3 back;
            if(face==2)back=wall(int(opt.x),hit.xz,hp,v,0.)*.83; // actual rear window/door, same material and room ID
            else if(face==4)back=mat(2,hit.xy/.8,hp,vec3(0,0,1),tt,nn,v,0.,.66);
            else if(face==5)back=vec3(.20,.20,.18)*irradiance;
            else { // exit through another outer pane: sky/ground, not a second apartment
                back=env(-v,.38)*.36;
                if((hit.z-lo.z)/(hi.z-lo.z)<low)back=vec3(.13,.14,.12)*irradiance;
            }
            float F=max(.16,.04+.96*pow(1.-max(dot(n,v),0.),5.));c=back*(1.-F)*.83+env(reflect(-v,n),.16)*F;
        }
    }else if(kind==5){vec2 s=uv/M(rid,1).xy;bool front=M(rid,1).z>.5;vec3 base=vec3(.66,.68,.66);if(front&&s.x>.08&&s.x<.92&&s.y>.15&&s.y<.87){vec2 q=(s-vec2(.5,.51))*vec2(1.7,1.);float rr=length(q);if(rr<.34){float blades=.5+.5*cos(atan(q.y,q.x)*7.+rr*12.);base=vec3(.08)+vec3(.12)*blades;if(rr<.09)base=vec3(.29);}if(fract(s.y*28.)<.19)base*=.49;}c=brdf(base,n,v,.57,.1,1.,world,direct);}
    else if(kind==6||kind==7){
        vec2 span=M(rid,1).xy;vec4 rect=kind==6?vec4(span.x*.19,max(.08,span.y-.70),span.x*.81,span.y-.12):
            vec4(max(.08,(span.x-1.)*.5),.04,min(span.x-.08,(span.x+1.)*.5),min(span.y-.10,1.95));
        if(rect.z>rect.x&&rect.w>rect.y&&contains(uv,rect))
            c=window(uv,rect,kind==6?1:4,a.w,world,t,b,n,v,direct,1.,8);
        else c=mat(layer,vec2(uv.x,-uv.y)/a.z,world,n,t,-b,v,direct,1.);
    }else if(kind==8){
        float stripe=mod(uv.x,.065);vec3 albedo=stripe<.021?vec3(.055,.058,.055):vec3(.23,.25,.24);
        c=brdf(albedo,n,v,.57,.45,1.,world,direct);
    }else{vec2 q=vec2(uv.x,-uv.y)/a.z;c=mat(layer,q,world,n,t,-b,v,direct,1.);}
    color=vec4(max(c,vec3(0)),1.);
}
