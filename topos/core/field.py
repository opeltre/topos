import torch
from topos.base import Cell
from .vect import Vect

class Field (Vect): 

    def __init__(self, domain, data, degree=0):
        self.domain = domain
        self.degree = degree
        self.data = data if isinstance(data, torch.Tensor)\
                else data * torch.ones([self.domain.size])

    def same(self, data=None):
        if isinstance(data, type(None)):
            data = self.data
        return self.domain.field(data)

    def get(self, a):
        a = self.domain[a] if not isinstance(a, Cell) else a
        return self.data[a.begin:a.end].view(a.shape.n)

    def __getitem__(self, a):
        return self.get(a)

    def __setitem__(self, a, va):
        a = self.domain[a] if not isinstance(a, Cell) else a
        try:
            if isinstance (va, torch.Tensor):
                data = va.reshape([a.size])
            else: 
                data = va * torch.ones([a.size])
            self.data[a.begin:a.end] = data
        except: 
            raise TypeError(f"scalar or size {a.size} tensor expected")

    def norm(self):
        return torch.sqrt((self.data ** 2).sum())

    def sum(self):
        return self.data.sum()

    #--- Show ---

    def __str__(self): 
        s = "{\n\n"
        for c in self.domain:
            sc = f"{c} ::"
            pad = len(sc) 
            s += sc + show_tensor(self.get(c), pad) + ",\n\n"
        return s + "}"
    
    def __repr__(self): 
        return f"{self.degree} Field {self}"

#--- Show --- 

def show_tensor (t, pad):
    return str(t).replace("tensor(", " " * 7)\
        .replace(")", "")\
        .replace("\n\n", "\n")\
        .replace("\t", "")\
        .replace(r'\s*', "")\
        .replace("\n", "\n" + " " * pad)
