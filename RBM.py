import topos 
import math
import torch

from math import prod
from torch.fft import fft, ifft

def tensor (x) : 
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)

def cylinder (n, js):
    slc = [None for k in range(n)]
    for j in js: 
        slc[j] = slice(None)
    return slc

def dot(u, *xs): 
    d, prod = u.dim(), torch.ones(u.shape)
    cs = [cylinder(d, [j]) for j in range(d)]
    prod *= u
    for xi, ci in zip(xs, cs):
        prod *= xi[ci]
    return prod.sum()

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

    S = full_simplex(*names)
    H0 = topos.Sheaf({
        f : join([shapes[x] for x in f]) for f in join(S) 
    })
    return H0

def DeepRBM (cells, shapes):

    sort  = lambda face : (len(face), *face)
    sortN = lambda chain : tuple(sort(f) for f in chain)

    RBMs = [RBM(c, *[shapes[i] for i in c.split(':')]) for c in cells]
    Z0   = topos.Union(*RBMs, sort=sort)

    G = topos.base.Poset(Z0.fibers.keys(), eltype=topos.base.Face)
    N1 = G.nerve(1)[1]
    N1.sort(key=sortN)
    Z1 = topos.Sheaf({c : Z0[c[-1]].shape for c in N1})

    return topos.Sum(Z0, Z1)

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
