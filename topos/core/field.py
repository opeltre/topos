import torch
from topos import Cell
from .vect import Vect

class Field (Vect): 

    def __init__(self, system, degree=0, data=0):
        self.system = system
        self.degree = degree
        self.domain = system[degree]
        self.data = data if isinstance(data, torch.Tensor)\
                else data * torch.ones([self.domain.size])

    def same(self, data=None):
        if isinstance(data, type(None)):
            data = self.data
        return self.domain.field(data)

    def is_same(self, other): 
        return  self.system == other.system\
            and self.degree == other.degree

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
