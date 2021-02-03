from itertools import product

from dict import Dict

class Set (set):
    """
    Hashable sets as strings.
    """
    def __init__(self, elems=(), sep=':'):
        self.sep = sep 
        if type(elems) == str:
            elems = elems.split(sep) if len(elems) > 0 else []
        super().__init__(elems)
    
    def __hash__(self): 
        return hash(str(self))

    def __str__(self): 
        elems = [str(e) for e in self]
        elems.sort()
        return "(" + self.sep.join(elems) + ")"

    def __repr__(self): 
        return f"Set {str(self)}"

    def __eq__(self, other): 
        return str(self) == str(other)
    
    def relate(self, r, arity=2):
        fibers = self ** (arity - 1)
        related = lambda a :\
            Set((b for b in fibers if r(a, *b))) if arity > 2 else\
            Set((b for b in self if r(a, b)))
        return Dict({a: related(a) for a in self})

    def graph(self, f): 
        return Set(((f(a), a) for a in self))        

    def fibers(self, f): 
        F = {}
        for y, x in self.graph(f): 
            F[y] = F[y] + [x] if y in F else [x]
        return Dict(F).fmap(Set)
    
    def __mul__(self, other): 
        return self.__class__(p for p in product(self, other))

    def __pow__(self, exp):
        return Set(p for p in product(self, repeat=exp))
