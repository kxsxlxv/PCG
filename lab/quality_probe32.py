"""Read the actual GLSL metric mask helpers in a float EGL framebuffer."""
from pathlib import Path
import numpy as np
from quant_probe import draw_probe

def band(p,r,aa):
    d=np.minimum(p-r[:2],r[2:]-p).min(axis=-1)
    t=np.clip(d/aa,0,1)
    return t*t*(3-2*t)

def probe(renderer):
    helper=(Path(__file__).parent/'material_lab'/'quality32.glsl').read_text()
    fragment='#version 330 core\n'+helper+'''
out vec4 color;
void main(){
    vec2 p=vec2(gl_FragCoord.x*.025-.20,gl_FragCoord.y*.025-.20);
    float sill=q32Sill(p,vec4(.2,.6,1.3,1.8),.02);
    float curtain=q32Curtain(p,1.,.02);
    float core=q32CorePanel(p,vec2(1.6,1.9),1.,.02);
    color=vec4(sill,curtain,core,1.);
}'''
    q=draw_probe(renderer,fragment,80,80)
    x=(np.arange(80)+.5)*.025-.2;y=x.copy()
    p=np.stack(np.meshgrid(x,y),axis=-1)
    sill=band(p,np.array([.155,.520,1.345,.592]),.02)
    curtain=np.maximum(band(p,np.array([.02,.015,.24,.985]),.02),band(p,np.array([.76,.015,.98,.985]),.02))
    core=band(p,np.array([.33,.06,1.27,1.71]),.02)
    expected=np.stack((sill,curtain,core),axis=-1)
    err=float(np.max(abs(q[:,:,:3]-expected)))
    assert err<.003,err
    inside=np.all(p>=[.2,.6],axis=-1)&np.all(p<=[1.3,1.8],axis=-1)
    assert np.all(q[:,:,0][inside]==0) and np.count_nonzero(q[:,:,0])>0
    return dict(samples=int(expected.size),max_error=err,aperture_overlap=0)
