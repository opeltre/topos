import torch 
from functools import reduce

from set import MapMixin
from dict import Dict

class VectorMixin(MapMixin): 

    def __add__(self, other):
        if type(other) in (int, float):
            return self.map(lambda ti, i: ti + other)
        D = self.domain() + other.domain()
        s = lambda i:\
            (self[i] if i in self else 0) +\
            (other[i] if i in other else 0)
        return self.__class__({i: s(i) for i in D})

    def __sub__(self, other): 
        return self.__add__(-1 * other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) in (int, float):
            return self.map(lambda tj, j: tj * other)
        s = self.map(lambda tj, j: other[j] if j in other else 0)
        return self.map(lambda tj, j: tj * s[j])
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self): 
        return f"Tensor {str(self)}"

    def __and__(self, other):
        return (self * other).sum()

    def sum(self):
        isTensor = lambda t:\
            isinstance(t, (torch.Tensor, VectorMixin))
        return sum(
            ti.sum() if isTensor(ti) else ti \
            for ti, i in self)

class MonadMixin:

    @classmethod 
    def unit(cls, a):
        return cls.__init__(a)

    def join(self):
        return self.__class__(
            reduce(lambda x, y: (*x, *y), self))

    def bind(self, f): 
        return self.fmap(f).join()


class Product (VectorMixin, tuple):
    
    def __iter__(self): 
        return ((xi, i) for i, xi in enumerate(super().__iter__()))
    
    def __str__(self): 
        elems = ' . '.join([str(xi) for xi, i in self])
        return f"{elems}"

    def __repr__(self): 
        return f"Product {str(self)}"

    def __or__(self, other): 
        return self.__class__(xi for xi, i in (*self, *other))

    def project(self, dim=(0,)):
        return self.restrict(dim)
    
    def flip(self, dim=(1, 0)):
        k = len(dim)
        return self.__class__(
            (self[dim[i]] if i < k else xi for xi, i in self))

    def fmap(self, f):
        return self.__class__((f(xi) for xi, i in self))

    def map(self, f):
        return self.__class__((f(xi, i) for xi, i in self))


class Vector (VectorMixin, Dict):
    pass


class Tensor (VectorMixin, Dict):  
    
    def __init__(self, d): 
        word = lambda a: a if isinstance(a, Product) else Product(a) 
        d = ({word(a): da for da, a in Dict(d)})
        super().__init__(d)

    def __or__(self, other): 
        t = {}
        for fa, a in self:
            for gb, b in other:
                t[a | b] = fa * gb
        return self.__class__(t)
        
    def sum(self, dim=None):
        if dim == None:
            return super().sum()
        proj = lambda ta, a: a.forget(dim)
        return self.fibers(proj).fmap(lambda tb: tb.sum())

    def curry(self, dim=0):
        proj = lambda ta, a: a.restrict(dim)
        t = self.fibers(proj).fmap(lambda tb: tb.sum(dim))
        return Operator(t)

    def flip(self, dim=(1,0)):
        t = lambda a: a.flip(dim)
        return self.__class__({
            t(a): fa for fa, a in self
        })


class Operator (Tensor): 

    def uncurry(self): 
        t = {}
        for ta, a in self:
            for tab, b in ta:
                t[a | b] = tab
        return Tensor(t)

    def transpose(self, dim=(1, 0)):
        return self.uncurry().flip(dim).curry()

    def t(self, dim=(1, 0)):
        return self.transpose(dim)

    def __matmul__(self, other): 
        if isinstance(other, Operator):
            B = other.t()
            AB = Tensor(self | B)
            return AB.fmap(lambda ab: ab.sum())
        return Tensor(A).map(lambda Ai, i: Ai & other[i])

    def __repr__(self): 
        return f"Operator {str(self)}"
