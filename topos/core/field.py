import torch
from topos.base import Fiber
from .vect import Vect

class Field (Vect): 

    def __init__(self, domain, data=0., degree=0):
        self.domain = domain
        self.degree = degree
        if isinstance(data, list):
            self.data = torch.tensor(data) 
        elif isinstance(data, torch.Tensor):
            self.data = data
        elif isinstance(data, (int, float)):
            self.data = data * torch.ones([self.domain.size])
        else:
            raise TypeError(
                f"unsupported data type: {type(data)}\n"\
                +"use: torch.Tensor, float, [float]...")

    def same(self, other=None):
        #--- Copy ---
        if isinstance(other, type(None)):
            return self.domain.field(self.data)
        #--- Create ---
        if not isinstance (other, Field):
            return self.domain.field(other)
        #--- Pass ---
        if self.domain == other.domain:
            return other
        #--- Extend ---
        if self.domain.scalars.size == other.domain.size:
            return self.domain.extend(other)

    def get(self, a):
        a = self.domain.get(a) if not isinstance(a, Fiber) else a
        return self.data[a.begin:a.end].view(a.shape.n)

    def __getitem__(self, a):
        return self.get(a)

    def __setitem__(self, a, va):
        a = self.domain[a] if not isinstance(a, Fiber) else a
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
