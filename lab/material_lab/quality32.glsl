// Stage 32 metric, bounded surface detail. Shared with the EGL probe.
float q32Band(vec2 p,vec4 rect,float aa){
    vec2 d=min(p-rect.xy,rect.zw-p);
    return smoothstep(0.,max(aa,.0001),min(d.x,d.y));
}
float q32Sill(vec2 p,vec4 opening,float aa){
    if(opening.z<=opening.x||opening.w<=opening.y)return 0.;
    return q32Band(p,vec4(opening.x-.045,opening.y-.080,opening.z+.045,opening.y-.008),aa);
}
float q32Curtain(vec2 q,float mode,float aa){
    if(mode<.5)return 0.;
    float left=q32Band(q,vec4(.02,.015,mode<1.5?.24:.42,.985),aa);
    float right=q32Band(q,vec4(mode<1.5?.76:.93,.015,.98,.985),aa);
    if(mode>2.5)return q32Band(q,vec4(.02,.48,.98,.985),aa);
    return max(left,right);
}
float q32CorePanel(vec2 s,vec2 span,float access,float aa){
    float width=min(access>.5?.94:1.28,span.x*.64);
    float z=access>.5?.06:span.y*.43;
    float height=min(access>.5?1.86:.63,span.y-z-.19);
    if(height<.20||width<.25)return 0.;
    return q32Band(s,vec4((span.x-width)*.5,z,(span.x+width)*.5,z+height),aa);
}
