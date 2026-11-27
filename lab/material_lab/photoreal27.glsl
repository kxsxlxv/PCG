// Stage 27 appearance reference. Shares stage-26 BRDF, metric POM and ray-box.
// The only new data is the separate appearance metadata columns and shared maps.
float phHash(float x){return fract(sin(x*12.9898+7.13)*43758.5453);}
float phNoise(vec2 p,float seed){
    vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);float k=i.x*127.1+i.y*311.7+seed*3.17;
    return mix(mix(phHash(k),phHash(k+127.1),f.x),mix(phHash(k+311.7),phHash(k+438.8),f.x),f.y);
}
bool phContains(vec2 s,vec4 r){return all(greaterThanEqual(s,r.xy))&&all(lessThan(s,r.zw));}
vec2 phWeights(float nv,float f0,float trans){
    float f=f0+(1.-f0)*pow(1.-clamp(nv,0.,1.),5.);float r=2.*f/(1.+f);
    return vec2(r,(1.-r)*trans);
}
vec3 phGlassProfile(int typ){return typ==2?vec3(.115,.36,.105):typ==1?vec3(.055,.60,.145):vec3(.043,.76,.115);}
vec3 phReflect(vec3 n,vec3 v,float rough){
    vec3 d=reflect(-v,n);vec3 c=env(d,rough);
    // Shared distant horizon proxy only (not real reflections of the test scene).
    // Continuous world direction, no per-window pseudo-random reflection map.
    float az=atan(d.y,d.x);float z=.047+.033*sin(az*13.)+.025*sin(az*29.+1.2);
    float horizon=(1.-smoothstep(z-.035-rough*.15,z+.035+rough*.15,d.z))*smoothstep(-.30,-.04,d.z);
    c=mix(c,vec3(.16,.21,.23),horizon*.65);
    c*=vec3(.91,.97,1.04);return c*(night>.5?.055:1.);
}
vec3 phGlass(vec3 interior,vec3 n,vec3 v,int typ,float rough,float dirt){
    vec3 g=phGlassProfile(typ);vec2 w=phWeights(max(dot(n,v),0.),g.x,g.y);
    vec3 tint=typ==2?vec3(.79,.92,.99):vec3(.94,.99,.97);
    return interior*w.y*tint*(1.-dirt*.12)+phReflect(n,v,rough)*w.x;
}
// Finite nearest ray/box hit; direction and metric coordinates are shared
// between both panes of a paired opening. No scene triangles or interior mesh.
void phBox(vec3 p,vec3 d,vec3 lo,vec3 hi,vec3 albedo,inout float best,inout vec3 c){
    vec3 safeD=mix(vec3(1e-6),d,greaterThan(abs(d),vec3(1e-6)));
    vec3 a=(lo-p)/safeD,b=(hi-p)/safeD;vec3 L=min(a,b),H=max(a,b);
    float t=max(max(L.x,L.y),L.z),e=min(min(H.x,H.y),H.z);
    if(t>=0.&&t<=e&&t<best){best=t;float top=float(abs(t-L.y)<.0001);c=albedo*(.75+.25*top);}
}
vec3 phInterior(vec2 q,vec2 ratio,vec2 aperture,float sill,float key,int semantic){
    // Room origin and dimensions are metric, not texture UVs stretched to each pane.
    float W=max(3.0,aperture.x+.85),H=max(2.72,aperture.y+sill+.32),D=semantic==2?5.4:semantic==1?4.6:3.7;
    vec3 p=vec3((W-aperture.x)*.5+q.x*aperture.x,sill+q.y*aperture.y,.015);
    vec3 d=normalize(vec3(-ratio,1.));vec4 hit=loggiaExit(p,d,vec3(0.),vec3(W,H,D));
    int face=int(hit.w);float best=length(hit.xyz-p);vec3 c;
    if(semantic==0){
        c=vec3(.37,.32,.25);if(face==2)c=vec3(.19,.13,.09)*(.86+.14*cos(hit.x*37.));
        else if(face==3)c=vec3(.59,.57,.51);else if(face==0||face==1)c*=.83;
        // Existing approved residential imagery, only on the rear surface.
        if(face==5&&roomSource>0){vec2 st=clamp(vec2(hit.x/W,1.-hit.y/H),.012,.988);c=texture(roomMaps,vec3(st,float(int(key)%8))).rgb;}
        else if(face==5)c*=.77+.23*phHash(key+32.);
        if(pixelsPerMeter*max(aperture.x,aperture.y)>25.){
            float sx=W*(phHash(key+4.)>.5?.64:.20);phBox(p,d,vec3(sx,0.,D*.62),vec3(min(sx+.84,W-.05),.71,D*.84),vec3(.23,.17,.12),best,c);
        }
        // Blinds/curtains at the room front; coherent across door/window pair.
        float curtain=phHash(key+71.);float edge=min(q.x,1.-q.x);
        if(curtain>.70&&edge<.13+.13*phHash(key+19.))c=mix(c,vec3(.42,.39,.33)*(.80+.20*cos(p.x*53.)),.93);
        if(curtain<.16){float slat=smoothstep(.22,.38,fract(p.y/.065));c=mix(c,vec3(.34,.32,.29)*(0.72+.28*slat),.82);}
        return c;
    }
    // Office and lobby NEVER sample the residential atlas. Distinct original
    // room layouts built from ray intersections with furniture and actual planes.
    c=semantic==1?vec3(.43,.45,.44):vec3(.50,.45,.38);
    if(face==2){float joints=step(.024,abs(fract(hit.x/.60)-.5))*step(.024,abs(fract(hit.z/.60)-.5));c=vec3(.23,.25,.24)*mix(.65,1.,joints);}
    if(face==3){float tile=step(.025,abs(fract(hit.x/.60)-.5))*step(.025,abs(fract(hit.z/.60)-.5));c=vec3(.64,.65,.62)*mix(.68,1.,tile);
        if(abs(fract(hit.x/1.5)-.5)<.09&&abs(fract(hit.z/1.8)-.5)<.35)c=vec3(1.,.98,.91);}
    if(face==5){float bands=step(.035,abs(fract(hit.x/1.1)-.5));c*=mix(.77,1.,bands);}
    if(pixelsPerMeter*max(aperture.x,aperture.y)>16.){
        if(semantic==1){
            float dx=.33+phHash(key+3.)*.28,zz=1.35+phHash(key+11.)*.45;vec3 desk=vec3(.34,.29,.23);
            phBox(p,d,vec3(dx,.73,zz),vec3(W-.40,.80,zz+.74),desk,best,c);
            phBox(p,d,vec3(dx+.08,.0,zz+.16),vec3(dx+.17,.73,zz+.66),vec3(.15,.16,.17),best,c);
            phBox(p,d,vec3(W-.59,.0,zz+.12),vec3(W-.47,.73,zz+.66),vec3(.15,.16,.17),best,c);
            float mx=W*.50-.28;phBox(p,d,vec3(mx,.88,zz+.40),vec3(mx+.57,1.23,zz+.46),vec3(.055,.064,.071),best,c);
            phBox(p,d,vec3(mx+.25,.79,zz+.40),vec3(mx+.31,.88,zz+.49),vec3(.13,.14,.15),best,c);
            phBox(p,d,vec3(mx+.05,.42,zz+1.00),vec3(mx+.56,.51,zz+1.47),vec3(.16,.20,.21),best,c);
            phBox(p,d,vec3(mx+.05,.51,zz+1.43),vec3(mx+.56,.96,zz+1.51),vec3(.16,.20,.21),best,c);
            if(phHash(key+38.)>.55)phBox(p,d,vec3(W-.43,.0,D-.85),vec3(W-.13,1.75,D-.22),vec3(.26,.28,.29),best,c);
        }else{
            phBox(p,d,vec3(.34,0.,D*.56),vec3(W*.67,1.02,D*.71),vec3(.37,.31,.23),best,c);
            phBox(p,d,vec3(.27,1.02,D*.54),vec3(W*.70,1.09,D*.74),vec3(.58,.55,.49),best,c);
            phBox(p,d,vec3(W*.75,0.,1.4),vec3(W-.18,.44,2.4),vec3(.19,.23,.22),best,c);
            phBox(p,d,vec3(.29,0.,D-.75),vec3(.67,.45,D-.38),vec3(.27,.20,.13),best,c);
            phBox(p,d,vec3(.23,.44,D-.82),vec3(.78,1.38,D-.30),vec3(.15,.26,.12),best,c);
        }
    }
    if(semantic==1&&phHash(key+51.)>.69){float slat=smoothstep(.18,.37,fract(p.y/.079));c=mix(c,vec3(.39,.40,.38)*(.72+.28*slat),.78);}
    return c;
}
vec3 phRoomLight(vec3 c,float state,int semantic){
    if(night<.5)return c*(semantic==2?.40:semantic==1?.23:.26);
    if(state<1.5)return c*.012;
    vec3 light=state<2.5?vec3(1.38,.86,.48):vec3(.91,1.02,1.11);
    return c*light*(semantic==2?1.32:semantic==1?1.05:1.18);
}
vec3 phEntry(vec2 q,vec2 size,int id,int role,vec3 p,vec3 t,vec3 up,vec3 n,vec3 v,float direct){
    vec3 base=M(id,13).xyz;int variant=int(M(id,10).z);vec2 m=q*size;
    float e=min(min(m.x,size.x-m.x),min(m.y,size.y-m.y));float edge=step(.055,e);
    float panel=1.;vec3 a;
    if(role==6){ // paneled timber door; no giant painted photograph
        float joint=min(abs(q.x-.22),abs(q.x-.78));float y=min(abs(q.y-.29),abs(q.y-.67));
        panel=(joint<.024||y<.014)?.65:1.;if(variant==1)panel=abs(fract(q.x*5.)-.5)<.035?.64:1.;
        if(variant==2)panel=abs(q.x-.50)<.035?.65:panel;
        float grain=.93+.07*sin(m.x*240.+sin(m.y*11.));a=base*panel*grain;
    }else{
        // Metal apartment door, coherent paint with localized wear and mail slot.
        a=base*(.91+.09*phNoise(m*vec2(4.,2.),M(id,8).w));
        if(variant==1&&q.y>.67&&q.y<.89&&abs(q.x-.5)<.30)a=mix(vec3(.08,.13,.15),phReflect(n,v,.28),.38);
        if(variant==2&&q.y>.14&&q.y<.84&&abs(q.x-.50)<.016)a*=.53;
        if(abs(q.y-.52)<.013&&abs(q.x-.50)<.16)a=vec3(.07,.08,.08);
    }
    if(edge<.5)a=mix(base*.52,M(id,9).xyz*.65,.35);
    float hx=variant==1?.23:.77;bool handle=abs(q.x-hx)<.026&&abs(q.y-.47)<.062;
    if(handle)return brdf(vec3(.42,.41,.35),n,v,.32,.70,1.,p,direct);
    return brdf(a,n,v,role==6?.69:.58,0.,edge<.5?.71:1.,p,direct);
}
vec3 phWindow(vec2 s,vec4 rect,vec4 roomRect,float roomSill,int role,float key,float state,int id,vec3 p,vec3 t,vec3 up,vec3 n,vec3 v,float direct){
    vec2 size=rect.zw-rect.xy,q=(s-rect.xy)/size;vec2 ratio=clamp(vec2(dot(v,t),dot(v,up))/max(dot(v,n),.10),-5.,5.);
    if(role==4||role==6)return phEntry(q,size,id,role,p,t,up,n,v,direct);
    int typ=int(M(id,8).y);int semantic=role==3?2:int(M(id,8).x)==2?1:0;int layout0=int(M(id,7).x)%4;
    bool door=role==2;int steps=pixelsPerMeter*max(size.x,size.y)>55.?20:pixelsPerMeter*max(size.x,size.y)>20.?8:0;
    vec4 hit=windowPOM(q,ratio,size,layout0,door,steps);int part=int(hit.w+.5);
    vec2 m=hit.xy*size;vec3 frame=M(id,9).xyz;float e=min(min(m.x,size.x-m.x),min(m.y,size.y-m.y));
    if(role==3){ // glazed double entrance leaves, transom and push bars
        part=e<.065||abs(q.x-.5)*size.x<.045||abs(q.y-.81)*size.y<.033?2:3;
        if(abs(q.y-.45)*size.y<.018&&abs(q.x-.5)>.085&&abs(q.x-.5)<.38)part=4;
        if(q.y<.065)part=5;
    }
    if(part==3){
        vec2 opening=roomRect.zw-roomRect.xy;vec2 rq=(rect.xy+hit.xy*size-roomRect.xy)/opening;
        vec3 c=phInterior(clamp(rq,0.,1.),ratio,opening,roomSill,key,semantic);c=phRoomLight(c,state,semantic);
        float dirty=M(id,8).z*exp(-m.y/.17);return phGlass(c,n,v,typ,M(id,9).w+dirty*.07,dirty);
    }
    vec3 a=part==1?frame*.63:part==4?vec3(.37,.40,.40):part==5?frame*.75:frame;
    float grain=.97+.03*sin(m.y*390.+sin(m.x*47.));a*=grain;if(part==2&&e>.083&&e<.094)a*=.32;
    return brdf(a,n,v,part==4?.37:.55,part==4?.40:0.,part==1?.67:.94,p,direct);
}
float phRectDistance(vec2 s,vec4 r){
    if(r.z<=r.x||r.w<=r.y)return 1e5;
    return length(max(max(r.xy-s,s-r.zw),vec2(0.)));
}
float phStain(vec2 s,vec4 r,float key){
    if(r.z<=r.x||r.w<=r.y||s.y>r.y)return 0.;
    float dx=max(max(r.x-s.x,s.x-r.z),0.);float v=r.y-s.y;
    float streak=.48+.52*phNoise(vec2(s.x*13.,s.y*.65),key);
    return exp(-dx/.045-v/.38)*streak;
}
vec3 phWall(int id,vec2 s,vec3 p,vec3 v,float direct){
    vec4 a=M(id,0),b=M(id,1),c=M(id,2),dims=M(id,6);vec3 t=M(id,4).xyz,n=M(id,5).xyz,up=vec3(0,0,1);
    int start=int(b.x),bays=int(b.y),floors=int(b.z),style=int(b.w),floorId=int(floor(s.y/max(c.z,.01))),bay=int(floor((s.x-c.x)/max(c.y,.01)+.5));
    vec4 rect=vec4(0.),other=vec4(0.),data=vec4(0.);int slot=start+floorId*bays+bay;
    bool valid=bays>0&&floorId>=0&&floorId<floors&&bay>=0&&bay<bays;
    if(valid){rect=O(slot,0);other=O(slot,1);data=O(slot,2);vec4 joint=rect;
        if(data.y>0.)joint=vec4(min(rect.xy,other.xy),max(rect.zw,other.zw));float sill=data.x==2?.17:dims.z;
        if(phContains(s,rect))return phWindow(s,rect,joint,sill,int(data.x),data.z,data.w,id,p,t,up,n,v,direct);
        if(phContains(s,other))return phWindow(s,other,joint,sill,int(data.y),data.z,data.w,id,p,t,up,n,v,direct);
    }
    int entry=start+bays*floors;vec4 en=O(entry,0);vec4 ed=O(entry,2);
    if(phContains(s,en))return phWindow(s,en,en,.05,int(ed.x),ed.z,ed.w,id,p,t,up,n,v,direct);
    float age=M(id,8).z,key=M(id,8).w;float rain=max(phStain(s,rect,key),phStain(s,other,key));
    float clearance=min(phRectDistance(s,en),min(phRectDistance(s,rect),phRectDistance(s,other)));
    // At floor/bay seams also check adjacent apertures; never paint a joint
    // through an enlarged paired block, its reveal clearance or entrance.
    if(bays>0){for(int fy=-1;fy<=1;fy++){for(int bx=-1;bx<=1;bx++){
        int f=floorId+fy,j=bay+bx;if(f<0||f>=floors||j<0||j>=bays||((fy==0)&&(bx==0)))continue;
        int k=start+f*bays+j;vec4 r0=O(k,0),r1=O(k,1);
        clearance=min(clearance,min(phRectDistance(s,r0),phRectDistance(s,r1)));
        rain=max(rain,max(phStain(s,r0,key),phStain(s,r1,key)));
    }}}
    float n0=phNoise(vec2(s.x*.22,s.y*.23),key),n1=phNoise(vec2(s.x*1.6,s.y*.41),key+17.);
    float tint=M(id,10).x*(1.+age*(n0-.50)*.16+age*(n1-.50)*.08);
    tint*=1.-age*(.13*rain+.17*exp(-max(s.y,0.)/.36));
    float floorZ=mod(max(s.y,0.),max(c.z,.01));int layer=int(a.y);float repeat=a.z;vec2 uv0=vec2(s.x,-s.y)/repeat;
    if(style==4&&bays>0&&clearance>.055){
        float du=abs(fract((s.x-c.x)/max(c.y,.01))-.5)*c.y;
        float dz=min(floorZ,c.z-floorZ);float seam=min(du,dz);
        float aa=max(.004,.50/max(pixelsPerMeter,1.));float j=1.-smoothstep(.007,.012+aa,seam);
        tint*=1.-j*min(1.,pixelsPerMeter*.036)*.32;
    }
    if(style==6&&mod(float(bay),4.)==3.)tint*=.83;
    if(style==7&&floorZ>dims.z-.08&&floorZ<dims.z+dims.w+.08)tint*=.60;
    if(style==9&&min(floorZ,c.z-floorZ)<.045)tint*=1.13;
    if(s.y<(style==8?.22:.46)){layer=35;uv0=vec2(s.x,-s.y)/1.6;tint*=.78;}
    else if(style==8&&floorZ>.14&&floorZ<c.z-.13){
        // Opaque curtain-wall spandrels, not additional transparent windows.
        vec3 panel=vec3(.105,.155,.185)*tint;return brdf(panel,n,v,.31,.27,1.,p,direct)+phReflect(n,v,.32)*.055;
    }
    return mat(layer,uv0,p,n,t,-up,v,direct,tint);
}
vec3 phSurface(int id,vec2 s,vec3 p,vec3 n,vec3 t,vec3 b,vec3 v,float direct){
    vec4 a=M(id,0),prop=M(id,8),q=M(id,12);int role=int(q.z),layer=int(a.y);
    vec2 xy=p.xy-q.xy;float age=prop.z;
    float tint=M(id,10).x*(1.+age*(phNoise(xy*.35,prop.w)-.5)*.12);vec2 st=vec2(s.x,-s.y)/a.z;
    if((role==1||role==2)&&layer>=33&&layer<=34){
        float r=floor(xy.y/1.05);float x=mod(xy.x+mod(r,2.)*2.1,4.2),y=mod(xy.y,1.05);
        float seam=min(min(x,4.2-x),min(y,1.05-y));tint*=mix(.73,1.,smoothstep(.012,.036,seam));
        float patch=phNoise(floor(xy/vec2(2.1,1.05)),prop.w+41.);if(patch>.83)tint*=1.13;
    }
    if(role==3){ // material/edge treatment variants of the existing canopy
        vec3 base=M(id,13).xyz;int k=int(M(id,10).z);vec3 col=k==0?mix(base,vec3(.48,.49,.45),.65):k==1?mix(base,vec3(.25,.28,.29),.72):vec3(.56,.57,.51);
        if(abs(n.z)<.2&&fract(s.x/.45)<.04)col*=.75;
        return brdf(col*tint,n,v,k==1?.42:.68,k==1?.15:0.,1.,p,direct);
    }
    if(role==4&&abs(n.z)<.2)return brdf(mix(M(id,11).xyz,vec3(.48,.46,.41),.50)*tint,n,v,.76,0.,1.,p,direct);
    if(role==5&&layer==7)return brdf(mix(M(id,9).xyz,vec3(.55,.53,.47),.55)*tint,n,v,.68,0.,1.,p,direct);
    if(role==8)return brdf(mix(M(id,9).xyz,vec3(.27,.30,.30),.68)*tint,n,v,.47,.35,1.,p,direct);
    return mat(layer,st,p,n,t,-b,v,direct,tint);
}
#ifndef PCG_PHOTOREAL_PROBE
void main(){
    vec4 a=M(rid,0);int kind=int(a.x),layer=int(a.y);vec3 n=normalize(gn),t=normalize(tangent),b=normalize(cross(n,t));vec3 v=perspective>.5?normalize(eye-world):normalize(viewDir);if(dot(n,v)<0.)n=-n;
    if(kind==3){vec4 c=texture(cutoutMaps,vec3(uv,float(layer)));
        if(c.a<.5)discard;if(passShadow==1){color=vec4(1);return;}
        vec3 col=layer==2?c.rgb:mix(M(rid,9).xyz,M(rid,11).xyz,.4)*(.82+.18*c.r);color=vec4(brdf(col,n,v,.59,.12,1.,world,sunVis(world,n)),1.);return;
    }
    if(passShadow==1){color=vec4(1);return;}
    float direct=sunVis(world,n);vec3 c;
    if(kind==1)c=phWall(rid,uv,world,v,direct);
    else if(kind==2){
        vec4 opt=M(rid,1);vec3 o=M(rid,2).xyz,tt=M(rid,3).xyz,nn=M(rid,4).xyz,lo=M(rid,5).xyz,hi=M(rid,6).xyz;vec2 span=M(rid,7).xy;
        vec3 p=vec3(dot(world-o,tt),dot(world-o,nn),world.z-o.z);vec3 d=-vec3(dot(v,tt),dot(v,nn),v.z);vec4 hit=loggiaExit(p,d,lo,hi);
        float low=opt.y,phase=uv.x/span.x,yy=uv.y/span.y;int pattern=int(M(rid,12).w);float spacing=pattern==4?.90:pattern==5?.52:.64;float count=max(1.,round(span.x/spacing));
        bool frame=min(phase,1.-phase)*span.x<.026||abs(phase*count-round(phase*count))*span.x/count<.024||abs(yy-low)*span.y<.025||(1.-yy)*span.y<.036;
        if(frame||yy<low){vec3 panel=frame?M(rid,9).xyz:M(rid,11).xyz;
            if(!frame&&pattern!=4)panel*=.88+.12*smoothstep(.04,.12,fract(uv.x/(pattern==5?.085:.15)));
            c=brdf(panel,n,v,.62,.03,.91,world,direct);
        }else{
            vec3 hp=o+tt*hit.x+nn*hit.y+vec3(0,0,hit.z);int face=int(hit.w+.5);vec3 back;
            if(face==2)back=phWall(int(opt.x),hit.xz,hp,v,0.)*.83;
            else if(face==4)back=mat(2,hit.xy/.8,hp,vec3(0,0,1),tt,nn,v,0.,.61);
            else if(face==5)back=vec3(.20,.20,.18)*irradiance;
            else back=phReflect(-n,v,.38)*.36;
            c=phGlass(back,n,v,0,.135,M(rid,8).z*.3);
        }
    }else if(kind==5){vec2 s=uv/M(rid,1).xy;bool front=M(rid,1).z>.5;vec3 base=vec3(.56,.58,.56);if(front&&s.x>.08&&s.x<.92&&s.y>.15&&s.y<.87){vec2 q=(s-vec2(.5,.51))*vec2(1.7,1.);float rr=length(q);if(rr<.34){float blades=.5+.5*cos(atan(q.y,q.x)*7.+rr*12.);base=vec3(.08)+vec3(.12)*blades;if(rr<.09)base=vec3(.29);}if(fract(s.y*28.)<.19)base*=.49;}c=brdf(base,n,v,.57,.1,1.,world,direct);}
    else if(kind==6||kind==7){vec2 span=M(rid,1).xy;vec4 rect=kind==6?vec4(span.x*.19,max(.08,span.y-.70),span.x*.81,span.y-.12):vec4(max(.08,(span.x-1.)*.5),.04,min(span.x-.08,(span.x+1.)*.5),min(span.y-.10,1.95));
        if(phContains(uv,rect))c=phWindow(uv,rect,rect,.65,kind==6?1:4,a.w,1.,rid,world,t,b,n,v,direct);
        else c=phSurface(rid,uv,world,n,t,b,v,direct);
    }else if(kind==8){float stripe=mod(uv.x,.065);c=brdf(stripe<.021?vec3(.055,.058,.055):vec3(.23,.25,.24),n,v,.57,.45,1.,world,direct);}
    else c=phSurface(rid,uv,world,n,t,b,v,direct);
    color=vec4(max(c,vec3(0)),1.);
}
#else
// Float framebuffer diagnostic uses the exact same Fresnel function as main.
void main(){int row=int(gl_FragCoord.y);float nv=float(int(gl_FragCoord.x))/32.;vec3 g=phGlassProfile(row);color=vec4(phWeights(nv,g.x,g.y),nv,1.);}
#endif
