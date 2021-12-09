from .hashable import Hashable
import torch

class Shape :

    def __init__(self, *ns):
        self.dim = len(ns)
        self.n = list(ns)
        self.ns = torch.tensor(self.n)
        self.mod = torch.tensor([
            torch.prod(self.ns[i+1:]) for i in range(self.dim)
        ])
        
        size = 1
        for ni in ns:
            size *= ni
        self.size = size

    def index(self, *js):
        if not len(js):
            return 0
        j0 = js[0]
        js = (j0 if isinstance(j0, torch.Tensor) and j0.dim() >= 1
                 else torch.tensor(js))
        return (self.mod * js).sum()

    def coords(self, i):
        if i >= self.size or i < 0:
            raise IndexError(f"{self} coords {i}")
        return torch.tensor([i]) % self.mod

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
        def emb_index (xs):
            ys = [0] * self.dim
            for d, x in zip(ds, xs):
                ys[d] = x
            return self.index(*ys)
        return emb_index
            

    def __iter__(self):
        return self.n.__iter__()
    
    def __str__(self): 
        return "(" + ",".join([str(ni) for ni in self.n]) + ")"

    def __repr__(self): 
        return f"Shape {self}"

