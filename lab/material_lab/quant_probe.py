"""Shared EGL framebuffer and texture quantization probes; no ModernGL dependency."""
import numpy as np
import gpu_lab as gpu

def draw_probe(renderer,fragment,w,h,uniforms=None):
    gl,C,I,U,P=gpu.gl,gpu.C,gpu.I,gpu.U,gpu.P
    vert='#version 330 core\nvoid main(){vec2 p=gl_VertexID==0?vec2(-1,-1):gl_VertexID==1?vec2(3,-1):vec2(-1,3);gl_Position=vec4(p,0,1);}'
    p=gl('glCreateProgram',U)()
    for typ,src in ((0x8B31,vert),(0x8B30,fragment)):gl('glAttachShader',None,U,U)(p,gpu.shader(typ,src))
    gl('glLinkProgram',None,U)(p);ok=I();gl('glGetProgramiv',None,U,U,P)(p,0x8B82,C.byref(ok));assert ok.value
    fb,_,_=renderer.framebuffer(w,h);gl('glUseProgram',None,U)(p);gl('glBindFramebuffer',None,U,U)(0x8D40,fb);gl('glViewport',None,I,I,I,I)(0,0,w,h)
    for name,i in (uniforms or {}).items():gl('glUniform1i',None,I,I)(gl('glGetUniformLocation',I,U,gpu.B)(p,name.encode()),i)
    gl('glDisable',None,U)(0x0B71);gl('glBindVertexArray',None,U)(renderer.vao);gl('glDrawArrays',None,U,I,I)(4,0,3)
    a=np.empty((h,w,4),np.float32);gl('glReadPixels',None,I,I,I,I,U,U,P)(0,0,w,h,0x1908,0x1406,gpu.ptr(a));assert gl('glGetError',U)()==0 and np.isfinite(a).all();assert np.all(a[:,:,3]==1.)
    gl('glUseProgram',None,U)(renderer.p);gl('glEnable',None,U)(0x0B71);gl('glDeleteProgram',None,U)(p)
    return a

def quant_probe(renderer,arrays):
    frag='''#version 330 core
uniform sampler2DArray albedoMaps,normalMaps,ormMaps;out vec4 color;
void main(){int x=int(gl_FragCoord.x),j=int(gl_FragCoord.y),k=x%16;ivec3 q=ivec3((k*73+19)%512,(k*127+13)%512,j);
vec3 c=x<16?texelFetch(albedoMaps,q,0).rgb:x<32?texelFetch(normalMaps,q,0).rgb:texelFetch(ormMaps,q,0).rgb;color=vec4(c,1.);}
'''
    a=draw_probe(renderer,frag,48,len(arrays[0]),{'albedoMaps':2,'normalMaps':3,'ormMaps':4});errors=[]
    for j in range(3):
        expect=np.stack([arrays[j][:,(k*127+13)%512,(k*73+19)%512,:3] for k in range(16)],axis=1);err=float(np.max(abs(a[:,16*j:16*(j+1),:3]-expect)));assert err<(.006 if j==0 else .0026),(j,err);errors.append(err)
    return dict(samples=int(len(arrays[0])*48),max_abs_error=errors,actual_formats=renderer.material_storage)

