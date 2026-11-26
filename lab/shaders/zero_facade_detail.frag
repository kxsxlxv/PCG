#version 330
in vec3 normal_ws;
in vec2 uv_m;
flat in uint record_id;
out vec4 frag;
uniform usampler2D param_u;
uniform sampler2D param_f;
uniform usampler2D cells;
uniform sampler2DArray base_color;
uniform sampler2DArray base_normal;
uniform sampler2DArray base_orm;
uniform sampler2DArray module_color;
uniform sampler2DArray module_nh;
uniform sampler2DArray module_mask;
uniform sampler2D macro_mask;
uniform vec3 view_dir;
uniform float daylight;
uniform float occupancy;
uniform int lane;
uniform int debug_mode;
uniform int dynamic_night;
const uint W=1024u;
const float PI=3.14159265;
ivec2 addr(uint i){return ivec2(int(i%W),int(i/W));}
uvec4 U(uint i){return texelFetch(param_u,addr(record_id*3u+i),0);}
vec4 F(uint i){return texelFetch(param_f,addr(record_id*4u+i),0);}
uint hash32(uint x){x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
float rnd(uint x){return float(hash32(x)&0xffffffu)/16777216.0;}
void main(){
    uvec4 a=U(0u),b=U(1u),c=U(2u);
    vec4 p=F(0u),q=F(1u),r=F(2u),s=F(3u);
    uint family=a.y,flags=a.z;
    uint surface_role=flags&3u;
    uint opening=0u,variant=0u;
    int floor_id=0,bay_id=0;
    vec2 local=vec2(0.0),dimensions=vec2(1.0);
    vec2 du=dFdx(uv_m),dv=dFdy(uv_m);
    if(surface_role==0u){
        if(b.y>0u){
            floor_id=int(floor(uv_m.y/p.z));
            bay_id=int(floor((uv_m.x-p.w)/q.x+0.5));
            if(floor_id>=0&&uint(floor_id)<b.x&&bay_id>=0&&uint(bay_id)<b.y){
                uint code=texelFetch(cells,addr(a.w+uint(floor_id)*b.y+uint(bay_id)),0).r;
                uint kind=code&3u;
                float center=p.w+float(bay_id)*q.x;
                float bottom=float(floor_id)*p.z+(kind==2u?r.y:q.w);
                dimensions=vec2(q.y,kind==2u?r.x:q.z);
                local=vec2((uv_m.x-center)/dimensions.x+0.5,(uv_m.y-bottom)/dimensions.y);
                if(kind>0u&&all(greaterThanEqual(local,vec2(0)))&&all(lessThan(local,vec2(1)))){
                    opening=kind;variant=code>>2u;
                }
            }
        }
        if((flags&16u)!=0u){
            vec2 dooruv=(uv_m-r.zw)/s.xy;
            if(all(greaterThanEqual(dooruv,vec2(0)))&&all(lessThan(dooruv,vec2(1)))){
                opening=3u;variant=0u;local=dooruv;dimensions=s.xy;
            }
        }
    }
    if(debug_mode==1){frag=vec4(float(record_id),uv_m,float(opening));return;}
    if(debug_mode==2){frag=vec4(float(record_id),float(family),float(variant),float(opening));return;}
    vec3 N=normalize(normal_ws),T=cross(vec3(0,0,1),N);
    if(dot(T,T)<1e-8)T=vec3(1,0,0)-N*N.x;
    T=normalize(T);vec3 B=cross(N,T);
    mat3 basis=mat3(T,B,N);
    vec3 V=normalize(view_dir),L=normalize(vec3(0.4,-0.6,0.7));
    vec2 base_uv=vec2(uv_m.x,-uv_m.y)/s.z;
    vec3 albedo=texture(base_color,vec3(base_uv,float(family))).rgb;
    float rough=0.82,metal=0.0,ao=1.0;
    bool detailed=lane!=0;
    bool enhanced=lane>=3;
    vec3 n=N;
    if(detailed){
        vec3 nm=texture(base_normal,vec3(base_uv,float(family))).rgb*2.0-1.0;
        n=normalize(basis*nm);
        vec3 orm=texture(base_orm,vec3(base_uv,float(family))).rgb;
        ao=orm.r;rough=orm.g;metal=orm.b;
    }
    if(surface_role==0u&&lane!=0){
        vec2 offset=vec2(float(c.x&65535u),float(c.y&65535u))/65536.0;
        float m=texture(macro_mask,uv_m/25.0+offset).r;
        albedo*=0.78+0.36*m+0.12*rnd(c.x);
        float dirt=s.w*0.28*exp(-max(uv_m.y,0.0)/0.75)*(0.35+0.65*m);
        albedo=mix(albedo,vec3(0.055,0.045,0.028),dirt);
        rough=mix(rough,0.94,dirt);
    }
    // Architectural trim in the existing shell, no extra pass.
    if(enhanced&&surface_role==0u&&opening==0u){
        float fw=max(length(du)+length(dv),0.002);
        float z=uv_m.y;
        float band=1.0-smoothstep(0.035,0.035+fw,abs(z-0.52));
        float top=1.0-smoothstep(0.10,0.10+fw,abs(z-(p.y-0.16)));
        float floorband=1.0-smoothstep(0.025,0.025+fw,abs(mod(z,p.z)-0.035));
        float plinth=1.0-smoothstep(0.47,0.47+fw,z);
        albedo=mix(albedo,albedo*0.58,plinth*0.72);
        albedo=mix(albedo,albedo*1.18+vec3(0.025),max(max(band,top),floorband*step(2.5,float(b.x))*0.65));
        if(b.y>0u){
            float center=p.w+floor((uv_m.x-p.w)/q.x+0.5)*q.x;
            float below=mod(z-q.w,p.z);
            int row=int(floor((z-q.w)/p.z))+1;
            int bay=int(floor((uv_m.x-p.w)/q.x+0.5));
            if(row>=0&&uint(row)<b.x&&bay>=0&&uint(bay)<b.y){
                uint code=texelFetch(cells,addr(a.w+uint(row)*b.y+uint(bay)),0).r;
                float runoff=exp(-abs(uv_m.x-center)/(0.27*q.y))*exp(-(p.z-below)/0.42);
                if((code&3u)>0u)albedo*=1.0-0.32*s.w*runoff;
            }
        }
    }
    float glass=0.0,curtain=1.0;
    if(opening!=0u){
        uint style=opening==3u?5u:opening==2u?4u:hash32(b.z)%4u;
        if(enhanced){
            style=opening==3u?16u+hash32(b.z)%2u:opening==2u?14u+hash32(b.z)%2u:6u+hash32(b.z)%8u;
        }
        vec2 mUV=vec2(local.x,1.0-local.y);
        vec2 dx=vec2(du.x/dimensions.x,-du.y/dimensions.y);
        vec2 dy=vec2(dv.x/dimensions.x,-dv.y/dimensions.y);
        vec4 mask=textureGrad(module_mask,vec3(mUV,float(style)),dx,dy);
        glass=opening==3u?0.0:mask.r;curtain=mask.g;
        albedo=textureGrad(module_color,vec3(mUV,float(style)),dx,dy).rgb;
        if(detailed){
            vec4 nh=textureGrad(module_nh,vec3(mUV,float(style)),dx,dy);
            n=normalize(basis*(nh.xyz*2.0-1.0));
            rough=mask.b;ao=mask.a;metal=0.0;
            if(lane==4&&glass>0.001){
                // Analytic room box intersection, no geometry or extra texture.
                vec3 vt=transpose(basis)*V;
                vec3 origin=vec3(local,0.0);
                vec3 ray=vec3(-vt.xy/max(abs(vt.z),0.15)/dimensions,1.0);
                float depth=1.35+0.65*rnd(b.z^uint(floor_id)*668265263u^uint(bay_id/2));
                vec2 side=mix(vec2(0.0),vec2(1.0),greaterThan(ray.xy,vec2(0)));
                vec2 t=(side-origin.xy)/(ray.xy+mix(vec2(-0.00001),vec2(0.00001),greaterThanEqual(ray.xy,vec2(0))));
                float hit=min(depth,min(t.x,t.y));
                vec3 pos=origin+max(hit,0.0)*ray;
                vec3 back=mix(vec3(0.065,0.055,0.043),vec3(0.19,0.17,0.13),rnd(b.w));
                vec3 room=back;
                if(t.x<min(depth,t.y))room*=0.64;
                else if(t.y<min(depth,t.x))room*=ray.y>0.0?1.28:0.47;
                else {
                    float panel=step(0.20,pos.x)*step(pos.x,0.72)*step(0.12,pos.y)*step(pos.y,0.42);
                    room=mix(room,room*0.43,panel);
                }
                float drape=smoothstep(0.18,0.47,abs(local.x-0.5))*curtain;
                room=mix(room,albedo*1.6,drape);
                albedo=mix(albedo,room,glass*0.8);
            }
            if(lane==2&&glass>0.001){
                vec2 shift=vec2(dot(V,T),-dot(V,B))*0.055*(1.0-nh.a)/
                    (max(abs(dot(V,N)),0.25)*dimensions);
                vec2 puv=clamp(mUV+shift,vec2(0.01),vec2(0.99));
                vec3 inside=textureGrad(module_color,vec3(puv,float(style)),dx,dy).rgb;
                albedo=mix(albedo,inside,glass);
            }
        }
    }
    vec3 emission=vec3(0);
    if(glass>0.0){
        uint h=b.z^uint(floor_id)*668265263u^uint(bay_id/2)*374761393u;
        uint wh=b.w^uint(floor_id)*2246822519u^uint(bay_id)*3266489917u;
        bool on=variant>0u;
        if(dynamic_night!=0){on=rnd(h)<occupancy;variant=(hash32(h^0x9e3779b9u)%4u)==0u?2u:1u;}
        if(on){
            vec3 lc=variant==2u?vec3(0.66,0.69,0.57):vec3(0.82,0.52,0.24);
            emission=lc*glass*curtain*(0.35+0.65*rnd(wh));
        }
    }
    if(debug_mode==3){frag=vec4(emission,float(opening));return;}
    if(!detailed){
        vec3 col=albedo*(0.68+0.32*max(dot(N,L),0.0))*daylight;
        frag=vec4(col+(1.0-daylight)*emission,1);return;
    }
    rough=clamp(rough,0.08,1.0);
    vec3 H=normalize(V+L);
    float nl=max(dot(n,L),0.0),nv=max(dot(n,V),0.001),nh=max(dot(n,H),0.0),vh=max(dot(V,H),0.0);
    float aa=rough*rough;aa*=aa;
    float D=aa/(PI*pow(nh*nh*(aa-1.0)+1.0,2.0));
    float k=pow(rough+1.0,2.0)/8.0;
    float G=(nl/(nl*(1.0-k)+k))*(nv/(nv*(1.0-k)+k));
    vec3 f0=mix(vec3(0.04),albedo,metal);
    vec3 fres=f0+(1.0-f0)*pow(1.0-vh,5.0);
    vec3 spec=D*G*fres/max(4.0*nl*nv,0.001);
    vec3 diffuse=(1.0-fres)*(1.0-metal)*albedo/PI;
    vec3 col=(diffuse+spec)*nl*2.1+albedo*ao*(0.30+0.12*max(n.z,0.0));
    if(glass>0.0){
        vec3 R=reflect(-V,n);
        vec3 env=mix(vec3(0.12,0.13,0.10),vec3(0.40,0.56,0.75),smoothstep(-0.20,0.35,R.z));
        float fr=0.04+0.96*pow(1.0-nv,5.0);
        col=mix(col,col*(1.0-fr)+env*fr,glass);
    }
    frag=vec4(col*daylight+(1.0-daylight)*emission,1);
}
