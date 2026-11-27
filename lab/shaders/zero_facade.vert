#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_layout;
in vec4 in_color;
uniform mat4 mvp;
out vec3 normal_ws;
out vec2 uv_m;
flat out uint record_id;
void main(){
    gl_Position=mvp*vec4(in_position,1.0);
    normal_ws=in_normal;
    uv_m=in_layout;
    uvec2 b=uvec2(round(in_color.rg*255.0));
    record_id=b.x+256u*b.y;
}
