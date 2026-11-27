// Stage 28: decorrelated metric appearance and bounded scene reflection proxy.
// All auxiliary resources are shared per tile; no per-building texture.
uniform sampler2D reflectionProxies;
float n28WorldNoise(vec3 p,float seed){
    return .44*phNoise(p.xy+vec2(.17,.43),seed)+.28*phNoise(p.xz+vec2(.31,.67),seed+11.)+.28*phNoise(p.yz+vec2(.71,.13),seed+29.);
}
// Finite slab intersection, rejects parallel rays outside the box.
float n28Hit(vec3 p,vec3 d,vec3 lo,vec3 hi){
    float a=0.,b=1e6;
    for(int k=0;k<3;k++){
        if(abs(d[k])<1e-6){if(p[k]<lo[k]||p[k]>hi[k])return -1.;}
        else{float s=(lo[k]-p[k])/d[k],t=(hi[k]-p[k])/d[k];a=max(a,min(s,t));b=min(b,max(s,t));}
    }
    return a<=b&&b>0.? (a>.001?a:b):-1.;
}
vec3 n28Reflect(vec3 n,vec3 v,float rough){
    vec3 d=normalize(reflect(-v,n));vec3 c=env(d,rough);
    // Direction-continuous cloud/ground contrast, not a per-window image.
    float blur=smoothstep(.12,.55,rough);
    float cloud=n28WorldNoise(d*5.7+vec3(3.2,1.8,.7),17.);
    cloud=smoothstep(.48-.12*blur,.65+.12*blur,cloud)*smoothstep(.015,.13,d.z);
    c=mix(c,vec3(.69,.73,.77),cloud*.66);
    float horizon=exp(-abs(d.z)/(.055+.12*rough));
    c=mix(c,vec3(.40,.47,.51),horizon*.30);
    // Four actual neighboring canonical footprint bounds. AABB silhouettes
    // are intentionally coarse; they are not screen-space or exact mesh rays.
    if(pixelsPerMeter>10.&&rough<.43){
        int bid=int(M(rid,10).w+.5);float best=1e6;vec3 col=vec3(0);float opacity=0.;vec3 origin=world+.025*n;
        for(int i=0;i<4;i++){
            vec4 L=texelFetch(reflectionProxies,ivec2(i*3,bid),0),H=texelFetch(reflectionProxies,ivec2(i*3+1,bid),0),A=texelFetch(reflectionProxies,ivec2(i*3+2,bid),0);
            if(L.w<.5)continue;float t=n28Hit(origin,d,L.xyz,H.xyz);
            if(t>.02&&t<best){best=t;vec3 p=origin+t*d;float fh=H.w;float floorZ=fract(p.z/max(fh,.5));
                float band=smoothstep(.06,.18,floorZ)*(1.-smoothstep(.72,.90,floorZ));float wall=.92-.30*band;
                col=A.rgb*wall*.78;opacity=exp(-t/260.)*(1.-blur*.70);}
        }
        if(best<1e5)c=mix(c,col,opacity*.88);
    }
    return c*vec3(.95,.99,1.02)*(night>.5?.072:1.);
}
uniform int externalPbrLayers;
vec3 n28MaterialTint(int layer,vec2 q,vec3 p,vec3 n,vec3 t,vec3 b,vec3 v,float direct,vec3 tint){
    float key=M(rid,8).w;vec3 a=texture(albedoMaps,vec3(q,float(layer))).rgb;
    vec3 orm=texture(ormMaps,vec3(q,float(layer))).rgb;
    vec3 tn=vec3(0,0,1);bool detailed=pixelsPerMeter>8.;
    if(detailed)tn=texture(normalMaps,vec3(q,float(layer))).rgb*2.-1.;
    bool authored=layer>=16&&layer<36&&((externalPbrLayers & (1 << (layer-16)))!=0);
    if(!authored&&layer>=22&&layer<=27&&pixelsPerMeter>6.){
        // Unstructured plaster/panel microtexture: independent phases with
        // smooth non-periodic metric weights. Never offset brick joints.
        float w=phNoise(q*.47+vec2(key*.019,key*.031),key+41.);
        vec2 shift=vec2(.39+.31*phHash(key+17.),.27+.46*phHash(key+43.));
        a=mix(a,texture(albedoMaps,vec3(q+shift,float(layer))).rgb,w);
        if(detailed){
            orm=mix(orm,texture(ormMaps,vec3(q+shift,float(layer))).rgb,w);
            tn=mix(tn,texture(normalMaps,vec3(q+shift,float(layer))).rgb*2.-1.,w);
        }
    }else if(!authored&&layer>=16&&layer<=18){
        float row=floor(q.y*16.);vec2 xy=vec2(q.x*5.+.5*mod(row,2.),q.y*16.);
        vec2 f=fract(xy);float edge=min(min(f.x,1.-f.x),min(f.y,1.-f.y));
        float unit=phHash(dot(floor(xy),vec2(71.3,217.7))+key*13.);
        a*=1.+(unit-.5)*.28*smoothstep(.045,.13,edge)*smoothstep(6.,26.,pixelsPerMeter);
    }else if(!authored&&layer>=19&&layer<=21){
        float cell=floor((layer==21?q.x:q.y)*8.);float unit=phHash(cell*71.3+key*13.);
        a*=1.+(unit-.5)*.19*smoothstep(5.,22.,pixelsPerMeter);
    }
    tn.xy*=.65;vec3 nn=normalize(t*tn.x+b*tn.y+n*max(.1,tn.z));
    return brdf(a*tint,nn,v,orm.g,orm.b,orm.r,p,direct);
}
vec3 n28Material(int layer,vec2 q,vec3 p,vec3 n,vec3 t,vec3 b,vec3 v,float direct,float tint){return n28MaterialTint(layer,q,p,n,t,b,v,direct,vec3(tint));}
// Existing ray/AABB helper, defined by the parent shader below this include.
void phBox(vec3 p,vec3 d,vec3 lo,vec3 hi,vec3 albedo,inout float best,inout vec3 c);
void n28Desk(vec3 p,vec3 d,float x,float X,float z,float Z,float key,inout float best,inout vec3 c){
    vec3 wood=mix(vec3(.26,.20,.14),vec3(.50,.45,.36),phHash(key));vec3 steel=vec3(.12,.14,.15);
    phBox(p,d,vec3(x,.72,z),vec3(X,.78,Z),wood,best,c);
    phBox(p,d,vec3(x+.06,0,z+.10),vec3(x+.11,.72,Z-.10),steel,best,c);
    phBox(p,d,vec3(X-.11,0,z+.10),vec3(X-.06,.72,Z-.10),steel,best,c);
    float m=(x+X)*.5;float q=z+.63*(Z-z);
    phBox(p,d,vec3(m-.28,.87,q),vec3(m+.28,1.22,q+.052),vec3(.07,.085,.10),best,c);
    phBox(p,d,vec3(m-.249,.898,q-.005),vec3(m+.249,1.188,q+.001),vec3(.16,.24,.31),best,c);
    phBox(p,d,vec3(m-.026,.78,q+.013),vec3(m+.026,.87,q+.037),steel,best,c);
    phBox(p,d,vec3(m-.22,.78,z+.09),vec3(m+.22,.801,z+.24),vec3(.13,.14,.15),best,c);
    phBox(p,d,vec3(m+.30,.78,z+.11),vec3(m+.39,.815,z+.22),vec3(.09,.11,.13),best,c);
    // Four physical chair surfaces, clear of the desk top.
    phBox(p,d,vec3(m-.24,.42,Z+.22),vec3(m+.24,.49,Z+.70),vec3(.18,.22,.23),best,c);
    phBox(p,d,vec3(m-.24,.49,Z+.64),vec3(m+.24,1.02,Z+.71),vec3(.16,.19,.20),best,c);
    phBox(p,d,vec3(m-.035,.08,Z+.44),vec3(m+.035,.42,Z+.51),steel,best,c);
}
vec3 n28Interior(vec2 q,vec2 ratio,vec2 aperture,float sill,float key,int semantic){
    float W=max(3.2,aperture.x+.85),H=max(2.78,aperture.y+sill+.32),D=semantic==2?5.8:semantic==1?4.8:3.8;
    vec3 p=vec3((W-aperture.x)*.5+q.x*aperture.x,sill+q.y*aperture.y,.015),d=normalize(vec3(-ratio,1.));
    vec4 hit=loggiaExit(p,d,vec3(0),vec3(W,H,D));int face=int(hit.w);float best=length(hit.xyz-p);vec3 c;
    float tone=.86+.28*phHash(key+31.);float detail=smoothstep(11.,40.,pixelsPerMeter*max(aperture.x,aperture.y));
    if(semantic==0){
        c=vec3(.45,.40,.32)*tone;
        if(face==2){float plank=floor(hit.x/.16);float shift=phHash(plank+key)*2.;float seam=min(abs(fract(hit.x/.16)-.5),abs(fract((hit.z+shift)/1.2)-.5));c=vec3(.24,.17,.10)*(.78+.18*phNoise(hit.xz*4.,key)+.16*smoothstep(.014,.04,seam));}
        else if(face==3)c=vec3(.62,.60,.55);
        else if(face==5&&roomSource>0)c=texture(roomMaps,vec3(clamp(vec2(hit.x/W,1.-hit.y/H),.012,.988),float(int(key)%8))).rgb;
        else if(face==0||face==1)c*=.83;
        if(detail>.30){
            float sx=phHash(key+8.)>.5?.28:W-1.10;
            phBox(p,d,vec3(sx,0.,D*.61),vec3(sx+.82,.69,D*.84),vec3(.29,.23,.17)*tone,best,c);
            phBox(p,d,vec3(sx,.69,D*.61),vec3(sx+.82,.74,D*.84),vec3(.41,.34,.25),best,c);
            phBox(p,d,vec3(sx+.28,.74,D*.68),vec3(sx+.40,1.05,D*.79),vec3(.48,.41,.29),best,c);
        }
        float curtain=phHash(key+71.);float edge=min(q.x,1.-q.x);
        if(curtain>.70&&edge<.10+.16*phHash(key+19.))c=mix(c,vec3(.45,.40,.33)*(.79+.21*cos(p.x*53.)),.93);
        if(curtain<.13)c=mix(c,vec3(.36,.34,.30)*(.70+.30*smoothstep(.23,.38,fract(p.y/.065))),.84);
        return c;
    }
    c=semantic==1?vec3(.47,.48,.46)*tone:vec3(.48,.43,.36)*tone;
    if(face==2){
        if(semantic==1)c=vec3(.19,.21,.21)*(.84+.18*n28WorldNoise(hit.xyz*4.,key));
        else{vec2 f=fract(hit.xz/.60);float seam=min(min(f.x,1.-f.x),min(f.y,1.-f.y));c=vec3(.32,.33,.31)*(.68+.32*smoothstep(.013,.032,seam));}
    }else if(face==3){
        vec2 f=fract(hit.xz/.60);float seam=min(min(f.x,1.-f.x),min(f.y,1.-f.y));c=vec3(.60,.62,.60)*(.77+.23*smoothstep(.018,.04,seam));
        float dx=abs(fract(hit.x/1.5)-.5),dz=abs(fract(hit.z/1.8)-.5);
        if(dx<.075&&dz<.36)c=mix(vec3(.85,.86,.81),vec3(1.,.98,.91),float(night>.5));
    }else if(face==5){float bands=step(.024,abs(fract(hit.x/1.2)-.5));c*=mix(.81,1.,bands);}
    if(detail>.27){
        int variant=int(phHash(key+38.)*3.);
        if(semantic==1){
            if(variant==0)n28Desk(p,d,.35,W-.42,1.45,2.17,key,best,c);
            else if(variant==1)n28Desk(p,d,.36,min(1.82,W-.50),1.85,2.62,key,best,c);
            else{ // meeting room: large table and chairs instead of identical desk
                phBox(p,d,vec3(.55,.73,1.57),vec3(W-.52,.80,3.15),vec3(.39,.33,.27),best,c);
                phBox(p,d,vec3(W*.48,0,2.22),vec3(W*.52,.73,2.34),vec3(.15,.16,.17),best,c);
                for(int j=0;j<2;j++){float x=.18+float(j)*(W-.65);phBox(p,d,vec3(x,.42,2.1),vec3(x+.45,.49,2.58),vec3(.18,.22,.24),best,c);phBox(p,d,vec3(x,.49,2.51),vec3(x+.45,.98,2.58),vec3(.16,.19,.20),best,c);}
            }
            // Low cabinets with individual drawer faces, away from aperture.
            float x=W-.48;phBox(p,d,vec3(x,0.,D-1.08),vec3(W-.12,1.15,D-.20),vec3(.32,.34,.33),best,c);
            for(int j=0;j<3;j++){float y=.09+float(j)*.34;phBox(p,d,vec3(x+.045,y,D-1.095),vec3(W-.165,y+.30,D-1.075),vec3(.42,.43,.40),best,c);}
        }else{
            float z=D*.57;phBox(p,d,vec3(.34,0.,z),vec3(W*.67,1.02,z+.65),vec3(.36,.30,.24),best,c);
            phBox(p,d,vec3(.27,1.02,z-.06),vec3(W*.70,1.09,z+.72),vec3(.55,.53,.49),best,c);
            phBox(p,d,vec3(W*.43,1.09,z+.18),vec3(W*.58,1.37,z+.23),vec3(.08,.10,.11),best,c);
            phBox(p,d,vec3(W*.75,0.,1.4),vec3(W-.18,.44,2.4),vec3(.19,.23,.22),best,c);
            phBox(p,d,vec3(W*.75,.44,2.32),vec3(W-.18,.91,2.43),vec3(.20,.24,.23),best,c);
            phBox(p,d,vec3(.29,0.,D-.75),vec3(.67,.45,D-.38),vec3(.27,.20,.13),best,c);
            for(int j=0;j<3;j++){float s=float(j)*.12;phBox(p,d,vec3(.30-s*.3,.44+s,D-.73+s*.4),vec3(.66+s*.3,1.12+s,D-.39+s*.4),vec3(.12,.23,.10)*(.87+s),best,c);}
        }
    }
    if(semantic==1&&phHash(key+51.)>.72){float slat=smoothstep(.18,.37,fract(p.y/.079));c=mix(c,vec3(.39,.40,.38)*(.72+.28*slat),.78);}
    return c;
}
vec3 n28Bounce(int id,vec3 lo,vec3 hi){
    if(night<.5)return vec3(0.);vec4 b=M(id,1),c=M(id,2);int bays=int(b.y),floors=int(b.z);
    int floorId=int(floor((lo.z+hi.z)*.5/max(c.z,.01))),bay=int(floor(((lo.x+hi.x)*.5-c.x)/max(c.y,.01)+.5));
    if(floorId<0||floorId>=floors||bay<0||bay>=bays)return vec3(0.);
    float state=O(int(b.x)+floorId*bays+bay,2).w;
    return state<1.5?vec3(0.):state<2.5?vec3(.045,.026,.012):vec3(.024,.033,.040);
}
