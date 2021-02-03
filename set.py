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
            Set((b for b in fibers if r(a, *b))) if arity > 2 else\
            Set((b for b in self if r(a, b)))
        return Dict({a: related(a) for a in self})

    def __str__(self): 
        elems = [str(e) for e in self]
        elems.sort()
        return "(" + self.sep.join(elems) + ")"

    def __repr__(self): 
        return f"Set {str(self)}"

