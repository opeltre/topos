from itertools import product

from dict import Dict
from mixin import Hashable, Mappable

class Set (Hashable, Mappable, set):
    """
    Hashable sets as strings.
    """
    def __init__(self, elems=(), sep=':'):
        self.sep = sep 
        if type(elems) == str:
            elems = elems.split(sep) if len(elems) > 0 else []
        super().__init__(elems)
    
    def relate(self, r, arity=2):
        fibers = self ** (arity - 1)
        related = lambda a :\
            self.__class__(
                (b for b in fibers if r(a, *b)) if arity > 2 else
                (b for b in self if r(a, b)))
        return Setmap({a: related(a) for a in self})

    def __str__(self): 
        elems = [str(e) for e in self]
        elems.sort()
        return "(" + self.sep.join(elems) + ")"

    def __repr__(self): 
        return f"Set {str(self)}"

    def __pow__(self, other): 
        if type(other) == int: 
            return super().__pow__(other)
        src = [x for x in other]
        values = self ** len(other)
        return values.fmap(lambda y: \
            Setmap({x: y[i] for i, x in enumerate(src)}))

class Setmap (Hashable, Mappable, Dict):

    def __str__(self):
        elems = [(k, ek) for ek, k in self]
        elems.sort()
        s = "{\n"
        for k, ek in elems: 
            s += f"{str(k)} :-> {ek}\n"
        s += "}"
        return s

    def __repr__(self):
        return f"Setmap {str(self)}" 

    def __mul__(self, other): 
        if not isinstance(Setmap, other): 
            return Set(self) * other
        fg = {}
        for (fx, x) in self:
            for (gy, y) in other:
                fg[(x, y)] = (fx, gy)
        return Setmap(fg)

    def __matmul__(self, other): 
        fog = {k: self[gk] for gk, k in other}
        return self.__class__(fog)
