import torch
import shape

class Field :

    def __init__(self, system, degree=0, data=None):
        self.system = system
        self.degree = degree
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.zeros([system.size[degree]])

    def get(self, a):
        a = self.system[a] if not isinstance(a, shape.Cell) else a
        return self.data[a.begin:a.end].view(a.shape.n)

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
    
    #--- Arithmetic Operations ---

    def same (self, data=None):
        if not isinstance(data, torch.Tensor):
            data = self.data
        return Field(self, self.system, self.degree, data)

    def __add__(self, other): 
        return self.same(self.data\
            + other.data if isinstance(other, Field) else other)

    def __sub__(self, other): 
        return self.same(self.data\
            - other.data if isinstance(other, Field) else other)

    def __mul__(self, other): 
        return self.same(self.data\
            * other.data if isinstance(other, Field) else other)

    def __div__(self, other): 
        return self.same(self.data\
            / other.data if isinstance(other, Field) else other)

    def __radd__(self, other): 
        return self.__add__(other)

    def __rsub__(self, other): 
        return self.__sub__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __iadd__(self, other):
        self.data += other.data if isinstance(other, Field) else other
        return self

    def __isub__(self, other):
        self.data -= other.data if isinstance(other, Field) else other
        return self

    def __imul__(self, other):
        self.data *= other.data if isinstance(other, Field) else other
        return self

    def __idiv__(self, other):
        self.data /= other.data if isinstance(other, Field) else other
        return self
