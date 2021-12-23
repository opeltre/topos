from .set import Set
from .simplex   import Chain, Seq
from .hashable  import Hashable

from itertools  import product

class Poset (Set):
    """ Partially ordered sets """

    def __init__(self, elems, sep=',', eltype=None, chains=None):

        self.eltype = eltype if eltype else Set

        if type(elems) == str:
            elems = elems.split(sep)
         
        K = [self.eltype.read(e) for e in elems]

        self.chains = [(a, b) for a, b in product(K, K) if a > b]\
                      if not chains else chains

        super().__init__(K, sep)
        
        self._below = {a : [] for a in self}
        self._above = {b : [] for b in self}
    
        for (a, b) in self.chains:
            self._below[a] += [b]
            self._above[b] += [a]

    def vertices(self): 
        vertices = Set()
        for face in self:
            for vertex in face:
                vertices.add(vertex)
        return vertices

    def below (self, a, strict=1):
        if not strict: 
            yield a
        for b in self._below[self.eltype.read(a)]:
            yield b

    def above (self, b, strict=1):
        if not strict: 
            yield b
        for a in self._above[self.eltype.read(b)]:
            yield a

    def between (self, a, c): 
        c = self.eltype.read(c)
        return (b for b in self.below(a) if b > c)

    def intercone (self, a, c, strict=1):
        c = self.eltype.read(c)
        return (b for b in self.below(a) if not b <= c)

    def nerve (self, degree = -1, sort=1, strict=1): 
        N = [[Chain(a) for a in self]]
        d = degree
        while d != 0:
            Nd = [Chain(*c, b) for c in N[-1] \
                               for b in self.below(c[-1], strict)] 
            d -= 1
            if len(Nd) == 0:
                break
            else:
                N += [Nd]
        if sort:
            def key(c)  : return tuple((-len(ci), str(ci)) for ci in c)
            for Nk in N : Nk.sort(key=key)
        return N

    def __mul__(self, other):
        elems = (Seq(a, b) for a, b in product(self, other))
        return Poset(elems, eltype=Seq)

    def __lt__(self, other): 
        for a in self:
            has_sup = False
            for b in other:
                if a <= b:
                    has_sup = True
                    break
            if not has_sup:
                return False
        return True

    def __repr__(self): 
        elems = [str(e) for e in self]
        s = ' '.join(elems)
        return f"Poset {s}"



class Hypergraph (Poset): 
    """
    Hypergraphs 
    """

    def __init__(self, elems, sep=',', **kwargs):
        super().__init__(elems, sep=',', eltype=Set) 

    def closure (self): 
        C = set()
        for a in self:
            for b in self: 
                c = Set(a & b)
                if c not in self:
                    C.add(c)
        if len(C) == 0:
            return self
        else:
            C = Hypergraph(C).closure()
            return Hypergraph(self | C)

    def item(self, other):
        return Set(other, self.sep)

    def __repr__(self): 
        elems = [str(e) for e in self]
        s = ' '.join(elems)
        return f"Hypergraph {s}"

