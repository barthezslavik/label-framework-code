# Independent check of the S^3 frame-charge lattice S(m,n) of paper2 (Thm 4.1).
# Output is in units of 4*pi^2. Run: uv run --with numpy python lattice_check.py

import numpy as np
rng=np.random.default_rng(0)
def qmul(a,b):
    a0,a1,a2,a3=a[...,0],a[...,1],a[...,2],a[...,3]; b0,b1,b2,b3=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3,a0*b1+a1*b0+a2*b3-a3*b2,a0*b2-a1*b3+a2*b0+a3*b1,a0*b3+a1*b2-a2*b1+a3*b0],-1)
def conj(a): return a*np.array([1,-1,-1,-1])
def qpow(x,m):
    r=np.zeros_like(x); r[...,0]=1
    y=x if m>=0 else conj(x)
    for _ in range(abs(m)): r=qmul(r,y)
    return r
def O(x,m,n):
    x=x/np.linalg.norm(x,axis=-1,keepdims=True)
    L=qpow(x,m); R=conj(qpow(x,n))
    E=np.eye(4)
    cols=[qmul(qmul(L,np.broadcast_to(E[k],x.shape)),R) for k in range(4)]
    return np.stack(cols,-1)  # O[...,A,B]
def S(m,n,N=400000,h=1e-5):
    x=rng.normal(size=(N,4)); x/=np.linalg.norm(x,axis=1,keepdims=True)
    # orthonormal tangent frame: x*i, x*j, x*k (right mult) -> oriented
    E=np.eye(4)
    t=[qmul(x,np.broadcast_to(E[k],x.shape)) for k in (1,2,3)]
    O0=O(x,m,n)
    dO=[(O(x+h*ti,m,n)-O(x-h*ti,m,n))/(2*h) for ti in t]
    e=lambda i: np.einsum('nab,nb->na',O0,t[i])          # e^A(t_i)
    de=lambda i,j: np.einsum('nab,nb->na',dO[i],t[j])-np.einsum('nab,nb->na',dO[j],t[i])
    val=sum((e(i)*de(j,k)).sum(1) for i,j,k in [(0,1,2),(1,2,0),(2,0,1)])
    return val.mean()*2*np.pi**2/(4*np.pi**2)
for m,n in [(0,0),(1,0),(2,0),(-1,0),(0,1),(0,2),(0,-1),(1,1),(3,0)]:
    print(m,n,round(S(m,n),3))
