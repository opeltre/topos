import topos 
import math
import torch

from math import prod
from torch.fft import fft, ifft

from time import time

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

class DeepRBM (topos.Sum):

    def __init__(self, cells, shapes): 

        sort  = lambda face : (len(face), *face)
        sortN = lambda chain : tuple(sort(f) for f in chain)

        RBMs = [RBM(c, *[shapes[i] for i in c.split(':')]) for c in cells]
        Z0   = topos.Union(*RBMs, sort=sort)
        Z    = []

        G = topos.base.Poset(Z0.fibers.keys(), eltype=topos.base.Face)
        N = G.nerve()
        for Nk in N: 
            Nk.sort(key=sortN)
            Z += [topos.Sheaf({c: Z0[c[-1]].shape for c in Nk})]
        
        self.scalars = topos.Sum(*Z)
        self.derived = topos.Sum(
                *[topos.Union(Zi, Zj) for Zi, Zj in zip(Z[:-1], Z[1:])])
        super().__init__(*Z)
        
        keys = []

        from time import time
        t0 = time()
        for k, fk in self[0].items():
            for i in range(fk.size):
                x = fk.shape.coords(i)
                kx = [f'{str(ki).lower()}{xi}' for ki, xi in zip(k[-1], x)]
                keys += [':'.join(kx)]
        t1 = time() 
        print(t1 - t0)
        self.fine = topos.System(keys)

        def Deff (U):
            pass

def evalRBM (f, *xs):
    RBM   = f.domain
    cells = list(RBM.fibers.keys())
    total = cells[-1]
    X = {i: tensor(xi) for i, xi in zip(total, xs)}
    dots = torch.stack([
        dot(f[a], *(X[i] for i in a)) for a in cells
    ])
    return dots.sum()

if __name__ == '__main__':
    K = DeepRBM(['X:Y', 'Y:Z'], {'X': [8], 'Y': [8], 'Z': [8]})

