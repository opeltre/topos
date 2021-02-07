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

    def __init__(self, f=1):
        if isinstance(f, (float, int, torch.Tensor)): 
            super().__init__({(Id,): f})
        elif isinstance(f, (dict, Tensor)):
            super().__init__(f)
        else:
            super().__init__({(f,) : 1})
    
    def __add__(self, other): 
        b = other if isinstance(other, Lambda) \
                else Lambda(other)
        return super().__add__(b) 

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other): 
        if isinstance(other, Lambda): 
            f = lambda x: self(x) * other(x)
            return Lambda(f)
        return super().__mul__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __call__(self, x): 
        return sum(
            ca * reduce(lambda p, Fj : p * Fj[0](x), Fa, 1)\
            for ca, Fa in self)

    def __str__(self): 
        s = "\n"
        for ca, Fa in self: 
            names = (fi.__name__ for fi, i in Fa)
            names = (n.replace("<lambda>", "\u03BB") for n in names)
            f = " * ".join(names)
            s += f": {ca} * \u03BB {f}\n"
        return s

    def __repr__(self): 
        s = "{" + str(self) + "}"
        return f"\u03BB {s}"

class Functional (Tensor): 
    
    def __init__(self, elems): 
        F = Dict(elems)
        coef = lambda Fi: isinstance(Fi, Lambda)\
            or not isinstance(Fi, (dict, Record))
        super().__init__({
            i : Lambda(Fi) if coef(Fi) else Functional(Fi)\
            for Fi, i in F
        })

    def curry(self, *args):
        return Operator(super().curry(*args))

    def __and__(self, other): 
        if isinstance (other, Functional): 
            return lambda t: sum(self[k](fk(t) for fk, k in other))
        return sum(self[k](tk) for tk, k in other)

    def sum(self): 
        isTensor = lambda t:\
            isinstance(t, (torch.Tensor, VectorMixin))
        return lambda t: Tensor(self.map(
                lambda fi, i: fi(t[i])
            )).sum()

    def __repr__(self): 
        return f"Functional {str(self)}"


class Operator (Functional, Matrix):
    pass
