from itertools import product

from dict import Dict

class Hashable:
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other): 
        return str(self) == str(other)

class Mappable: 
    def fmap(self, f):
        return self.__class__((f(x) for x in self))

    def graph(self, f): 
        return self.__class__((f(a), a) for a in self)

    def fibers(self, f): 
        F = {}
        for y, x in self.graph(f): 
            F[y] = F[y] + [x] if y in F else [x]
        return Dict(F).fmap(self.__class__)

    def __mul__(self, other): 
        return self.__class__(p for p in product(self, other))

    def __pow__(self, other):
        if type(other) == int:
            return self.__class__(p for p in product(self, repeat=other))
        src = enumerate([x for x in other])
        functions = product((self for x in src))
        return self.__class__(
            (Dict({x: f[i] for i, x in src }) for f in functions))
            
