// Reusable GLSL 330 reference: host-space ray exit and metric window depth.
// Local coordinates: X facade tangent, Y outward, Z up.
vec4 loggiaExit(vec3 p, vec3 d, vec3 lo, vec3 hi) {
    p=clamp(p,lo,hi);
    vec3 ts=vec3(1e20);
    for(int i=0;i<3;i++) if(abs(d[i])>1e-8) {
        ts[i]=((d[i]>0.0?hi[i]:lo[i])-p[i])/d[i];
    }
    int a=0;
    if(ts.y<ts.x) a=1;
    if(ts.z<ts[a]) a=2;
    int face=2*a+(d[a]>0.0?1:0);
    return vec4(clamp(p+ts[a]*d,lo,hi),float(face));
}
// Height profile in metric coordinates; depth is positive INTO the wall.
// Outer reveal, frame, sash, seals, sill and glass are distinct surfaces.
vec2 windowDepth(vec2 p,vec2 size,int windowLayout,bool door){
    float e=min(min(p.x,size.x-p.x),min(p.y,size.y-p.y));
    if(e<0.0) return vec2(0.,1.);
    if(p.y<.036) return vec2(.016,4.);
    if(e<.052) return vec2(.025+.65*e,1.);
    if(e<.094) return vec2(.065,2.);
    float bar=1e6;
    if(windowLayout==0)bar=abs(p.x-size.x*.5);
    if(windowLayout==1)bar=abs(p.x-size.x*.38);
    if(windowLayout==2)bar=min(abs(p.x-size.x/3.),abs(p.x-2.*size.x/3.));
    if(windowLayout==3)bar=min(abs(p.x-size.x*.5),abs(p.y-size.y*.72));
    if(door)bar=abs(p.y-size.y*.30);
    if(bar<.027)return vec2(.073,2.);
    if(door&&p.y<size.y*.30)return vec2(.09,5.);
    return vec2(.125,3.);
}
vec4 windowPOM(vec2 uv,vec2 ratio,vec2 size,int windowLayout,bool door,int steps){
    vec2 p=uv*size;
    if(steps<=0){vec2 h=windowDepth(p,size,windowLayout,door);return vec4(uv,h);}
    float delta=.14/float(steps);float prev=windowDepth(p,size,windowLayout,door).x;
    for(int i=1;i<=20;i++){
        if(i>steps)break;
        float z=float(i)*delta;
        vec2 q=p-ratio*z;vec2 h=windowDepth(q,size,windowLayout,door);
        if(z>=h.x){
            float a=prev-(z-delta);float b=z-h.x;
            float f=clamp(a/max(a+b,1e-8),0.,1.);
            float depth=z-delta+delta*f;
            return vec4(clamp((p-ratio*depth)/size,0.,1.),depth,h.y);
        }
        prev=h.x;
    }
    return vec4(clamp((p-ratio*.14)/size,0.,1.),.14,3.);
}
