import torch
from topos import Cell
from vect import Vect

class Field (Vect): 

    def __init__(self, system, degree=0, data=None):
        self.system = system
        self.degree = degree
        data = data if isinstance(data, torch.Tensor)\
                else torch.zeros([system.size[degree]])
        super().__init__(system.size[degree], data)

    def get(self, a):
        a = self.system[a] if not isinstance(a, Cell) else a
        return self.data[a.begin:a.end].view(a.shape.n)

    def same(self, data=None):
        if isinstance(data, type(None)):
            data = self.data
        return Field(self.system, self.degree, data)

    def is_same(self, other): 
        return  self.system == other.system\
            and self.degree == other.degree

    #--- Show ---

    def __str__(self): 
        def tensor (t, pad): 
            return str(t).replace("tensor(", " " * 7)\
                    .replace(")", "")\
                    .replace("\n\n", "\n")\
                    .replace("\t", "")\
                    .replace(r'\s*', "")\
                    .replace("\n", "\n" + " " * pad)
        s = "{\n\n"
        for c in self.system.nerve[self.degree]:
            sc = f"{c} ::"
            pad = len(sc) 
            s += sc + tensor(self.get(c), pad) + ",\n\n"
        return s + "}"
    
    def __repr__(self): 
        return f"{self.degree}-Field {self}"
    
