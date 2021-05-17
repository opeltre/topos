from .set import Set
from .tensor import Tensor

class Hypergraph (Set): 
    """
    Hypergraphs 
    """
    def __init__(self, elems, sep=','):
        if type(elems) == str:
            elems = elems.split(sep)
        K = (Set(e) if type(e) == str else e for e in elems)
        super().__init__(K, sep)
        
        self.nerve = []

    def vertices(self): 
        vertices = Set()
        for face in self:
            for vertex in face:
                vertices.add(vertex)
        return vertices

    def below (self, a, strict=1):
        if not strict: 
            yield a
        for b in self._below[a]:
            yield b

    def above (self, b, strict=1):
        return self._above[b]

    def between (self, a, c): 
        return self.below(a).above(c)

    def intercone (self, a, b, strict=1):
        icone = self.below(a, strict).difference(self.below(b))
        return Hypergraph(icone)

    def add(self, elem): 
        super().add(Set((elem)))

    def close(self): 
        """ /!\ mutable """
        self |= self.closure()
        return self

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

