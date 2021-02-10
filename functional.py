import torch
from functools import reduce

from tensor import VectorMixin, Tensor, Matrix

from set import MapMixin
from dict import Dict, Record

def Id (x):
    return x

def One (x):
    return 1

class Lambda (Tensor):

    def __init__(self, f=1, name='\u03BB'):
        self.__name__ = name
        if isinstance(f, (float, int, torch.Tensor)): 
            super().__init__({(Id,): f})
            self.__name__ = str(f)
        elif isinstance(f, (dict, Tensor)):
            super().__init__(f)
        elif callable(f):
            if '__name__' in f.__dir__(): 
                self.__name__ = f.__name__
            super().__init__({(f,) : 1})
    
    def __call__(self, x): 
        return sum(
            ca * reduce(lambda p, Fj : p * Fj[0](x), Fa, 1)\
            for ca, Fa in self)

    def __matmul__(self, other): 
        fog = lambda t: self(other(t))
        fog.__name__ = f"{f.__name__} o {g.__name__}"
        return fog

    def __add__(self, other): 
        b = other if isinstance(other, Lambda) \
                else Lambda(other)
        return super().__add__(b) 

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other): 
        if isinstance(other, Lambda): 
            return super().__or__(other)
        return super().__mul__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def sum(self, *args): 
        return self

    def __repr__(self): 
        return str(self)

    def __str__(self): 
        s = ""
        l = "\u03BB: "
        for ca, Fa in self: 
            names = ((fi.__name__ if '__name__' in fi.__dir__() else '_')\
                    for fi, i in Fa)
            names = [n.replace("<lambda>", "\u03BB") for n in names]
            f = " * ".join(names)
            s += f"{ca} * {f}\n"
        return l + ('\n' + s).replace('\n', '\n  ') \
            if len(self.domain()) > 1 else \
            '\t' + l + s.replace('\n', '')

class Functional (Tensor): 
    
    def __init__(self, elems): 
        F = Record(elems)
        coef = lambda Fi: isinstance(Fi, Lambda)\
            or not isinstance(Fi, (dict, Record, Dict))
        super().__init__({
            i : Lambda(Fi) if coef(Fi) else Functional(Fi)\
            for Fi, i in F
        })

    def __radd__(self, other): 
        return self.__add__(other) 

    def __and__(self, other): 
        return Lambda({
            (Fi @ other[i], ): 1 for Fi, i in self
        })

    def __repr__(self): 
        s = "{\n" + str(self) + "}"
        return f"\u03BB-Tensor {s}"


class Operator (Functional, Matrix):
    pass
