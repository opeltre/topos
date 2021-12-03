import topos 
import math
import torch

from math import prod
from torch.fft import fft, ifft

def full_simplex (*names):
    S = topos.base.Face(*names)
    F = [[(0, S)]]
    def nextf (nfaces):
        faces = []
        for i, f in nfaces:
            faces += [(j, f.d(j)) for j in range(i, f.degree + 1)]
        return faces
    for d in range(S.degree + 1):
        F += [nextf(F[-1])]
    return [[f for i, f in Fd[::-1]] for Fd in F[::-1]]

def RBM (names, *shapes):

    def join (ll): return [x for l in ll for x in l]

    names = (names if not isinstance(names, str)
                   else names.split(":"))
    shapes = (shapes if isinstance(shapes, dict)
                     else {x: s for x, s in zip(names, shapes)})

    K = full_simplex(*names)
    return topos.Sheaf({
        f : join([shapes[x] for x in f]) for f in join(K)
    })
       
def tensor (x) : 
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)

def dot(u, *xs): 
    d, prod = u.dim(), torch.ones(u.shape)
    prod *= u
    for i, x in enumerate(xs):
        _i_  = ([None for k in range(i)] 
             + [slice(None)]            
             + [None for k in range(i+1, d)])
        prod *= x[_i_]       
    return prod.sum()

def evalRBM (f, *xs):
    RBM   = f.domain
    cells = list(RBM.fibers.keys())
    total = cells[-1]
    X = {i: tensor(xi) for i, xi in zip(total, xs)}
    dots = torch.stack([
        dot(f[a], *(X[i] for i in a)) for a in cells
    ])
    return dots.sum()

#-- Fourier ---

def Fourier (N):
    kx = torch.tensor([ k * x for k in range(N) for x in range(N)])
    F  = torch.exp(2j * math.pi * kx / N)
    return F.reshape([N, N]) / math.sqrt(N)

F = Fourier(2)
x = torch.tensor([-1, 1]).cfloat()
