from set import Set
from simplex import Simplex

class Hypergraph (Set): 
    """
    Hypergraphs 
    """
    def __init__(self, elems, sep=','):
        if type(elems) == str:
            elems = elems.split(sep)
        elems = (Set(e) for e in elems)
        super().__init__(elems, sep)

    def vertices(self): 
        vertices = Set()
        for face in self:
            for vertex in face:
                vertices.add(vertex)
        return vertices

    def add(self, elem): 
        super().add(Set((elem)))

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
            C = Hypergraph(C).close()
            return Hypergraph(self | C)

    def close(self): 
        if not len(self): 
            return self
        C = Hypergraph(())
        for a in self:
            for b in self: 
                c = Set(a & b)
                if c not in self:
                    C.add(c)
        self |= C.close()
        return self

    def below (self, a):
        cone = set()
        for b in self: 
            if a > b: 
                cone.add(b)
        return Hypergraph(cone)

    def above (self, b):
        cone = set()
        for a in self: 
            if a > b: 
                cone.add(a)
        return Hypergraph(cone)

    def between (self, a, c): 
        return self.below(a).above(c)

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

    def nerve(self, n):
        if n == 0:
            return [Simplex([a]) for a in self]
        N = [] 
        for c in self.nerve(n-1):
            cofaces = [Simplex([a] + c[:]) for a in self.above(c[0])]
            N += cofaces
        return N

    def __repr__(self): 
        elems = [str(e) for e in self]
        s = ' '.join(elems)
        return f"Hypergraph {s}"

    def item(self, other):
        return Set(other, self.sep)
