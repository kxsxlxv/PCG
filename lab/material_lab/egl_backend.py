"""Run actual GLSL via EGL/GL using system libraries; no ModernGL dependency.
Float output shader probe, not a production PBR renderer or performance test.
"""
import ctypes as C, ctypes.util, os, sys, json
from pathlib import Path
import numpy as np

os.environ.setdefault('EGL_PLATFORM','surfaceless');os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
E=C.CDLL(ctypes.util.find_library('EGL'));G=C.CDLL(ctypes.util.find_library('GL'))
P=C.c_void_p;I=C.c_int;U=C.c_uint;F=C.c_float;B=C.c_char_p

def bind(lib,name,restype,*args):
    f=getattr(lib,name);f.restype=restype;f.argtypes=list(args);return f

def egl(name,ret,*args):return bind(E,name,ret,*args)
def gl(name,ret,*args):return bind(G,name,ret,*args)
def ints(*a):return (I*len(a))(*a)
def ptr(a):return P(a.ctypes.data)
D=egl('eglGetDisplay',P,P)(None);major=I();minor=I();assert egl('eglInitialize',U,P,P,P)(D,C.byref(major),C.byref(minor));assert egl('eglBindAPI',U,U)(0x30A2)
config=P();n=I();assert egl('eglChooseConfig',U,P,P,P,I,P)(D,ints(0x3033,1,0x3040,8,0x3024,8,0x3023,8,0x3022,8,0x3038),C.byref(config),1,C.byref(n)) and n.value
surf=egl('eglCreatePbufferSurface',P,P,P,P)(D,config,ints(0x3057,1,0x3056,1,0x3038));ctx=egl('eglCreateContext',P,P,P,P,P)(D,config,None,ints(0x3098,3,0x30FB,3,0x30FD,1,0x3038));assert ctx and surf
assert egl('eglMakeCurrent',U,P,P,P,P)(D,surf,surf,ctx)
print('renderer',gl('glGetString',B,U)(0x1F01).decode())

def shader(typ,code):
    s=gl('glCreateShader',U,U)(typ);b=B(code.encode());gl('glShaderSource',None,U,I,P,P)(s,1,C.byref(b),None);gl('glCompileShader',None,U)(s);v=I();gl('glGetShaderiv',None,U,U,P)(s,0x8B81,C.byref(v));
    if not v.value:
        log=C.create_string_buffer(8192);gl('glGetShaderInfoLog',None,U,I,P,P)(s,len(log),None,log);raise RuntimeError(log.value.decode())
    return s

def gen(name):
    x=U();gl(name,None,I,P)(1,C.byref(x));return x.value

