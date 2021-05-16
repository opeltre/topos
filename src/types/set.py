from itertools import product, chain

from map import Map
from hashable import Hashable

class Iterable: 

    def fmap(self, f):
        return self.__class__((f(x) for x in self))

    def foldl(self, f, acc=None):
        for x in self:
            acc = f(acc, x) if not isinstance(acc, NoneType)\
                            else x

    def filter(self, f): 
        return self.__class__((x for x in self if f(x)))

    def fibers(self, f): 
        F = {}
        for x in self: 
            y = f(x)
            F[y] = F[y] + [x] if y in F else [x]
        return F

    def __add__(self, other): 
        return self.__class__(s for s in chain(self, other))

    def __mul__(self, other): 
        return self.__class__(p for p in product(self, other))

    def __pow__(self, n):
        return self.__class__(p for p in product(self, repeat=n))

    def __str__(self): 
        elems = [str(e) for e in self]
        elems.sort()
        return "(" + self.sep.join(elems) + ")"


class Set (Iterable, Hashable, set):

    def __init__(self, elems=(), sep=':'):
        self.sep = sep 
        if type(elems) == str:
            elems = elems.split(sep) if len(elems) > 0 else []
        super().__init__(elems)
    
    def __and__(self, other):
        cap = super().__and__(other)
        return self.__class__(cap)

    def __or__(self, other):
        cup = super().__or__(other)
        return self.__class__(cup)

    def __sub__(self, other): 
        return self.__class__(self.difference(other))

    def fibers(self, f):
        F = super().fibers(f)
        return Map(F).fmap(Set)

    def __repr__(self): 
        return f"Set {str(self)}"

    def __pow__(self, other): 
        if type(other) == int: 
            return super().__pow__(other)
        src = [x for x in other]
        values = self ** len(other)
        return values.fmap(lambda y: \
            Map({x: y[i] for i, x in enumerate(src)}))
