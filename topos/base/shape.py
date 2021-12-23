from .hashable import Hashable
import torch

class Shape (Hashable):

    def __init__(self, *ns):
        self.dim = len(ns)
        self.n = list(ns)
        self.ns = torch.tensor(self.n)
        self.mod = torch.tensor([
            torch.prod(self.ns[i+1:]) for i in range(self.dim)
        ])
        self.rmod = torch.tensor([
            torch.prod(self.ns[:-i]) for i in range(self.dim)
        ])
        
        size = 1
        for ni in ns:
            size *= ni
        self.size = size

    def index(self, *js):
        """ Row-major index of coordinates js.

            If a 2D tensor is given as coordinates,
            the first dimension is understood as batch
            dimension.
        """
        if not len(js):
            return 0
        j0 = js[0]
        js = (j0 if isinstance(j0, torch.Tensor) and j0.dim() >= 1
                 else torch.tensor(js))
        return (self.mod * js).sum([-1]) 

    def coords(self, i):
        """ Returns coordinates of row-major index.

                i :: int | tensor(dtype=long)
        """
        if isinstance(i, int):
            if i >= self.size or i < 0:
                raise IndexError(f"{self} coords {i}")

        div = lambda a, b: torch.div(a, b, rounding_mode = 'floor')
        
        i = (torch.tensor(i) if not isinstance(i, torch.Tensor)
             else i + 0)
        
        if i.dim() == 0:
            out = torch.zeros([self.dim], dtype=torch.long)
            for j, mj in enumerate(self.mod):
                out[j] = div(i, mj)
                i      = i % mj
            return out

        out = []
        for j, mj in enumerate(self.mod):
            out += div(i[None,:], mj)
            i    = i % mj
        return (torch.stack(out).t() if len(out) 
                                     else torch.tensor([[]]))
            

    def p(self, d):
        def proj_d(i):
            x = self.coords(i)
            return x[d]
        return proj_d

    def res(self, *ds):
        def res_index(i):
            x   = self.coords(i)
            tgt = Shape(*[self.n[d] for d in ds])
            return tgt.index(*[x[d] for d in ds])
        return res_index

    def embed(self, *ds):
        mat = torch.zeros([self.dim, len(ds)], dtype=torch.long)
        for i, d in enumerate(ds):
            mat[d, i] = 1
        def emb_index(x):
            y = (mat @ x if x.dim() == 1. 
                         else (mat @ x.T).T)
            return int(self.index(y))
        return emb_index

    def __iter__(self):
        return self.n.__iter__()
    
    def __str__(self): 
        return "(" + ",".join([str(ni) for ni in self.n]) + ")"

    def __repr__(self): 
        return f"Shape {self}"

