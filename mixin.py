from itertools import product, chain

from dict import Dict


class Hashable:

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other): 
        return str(self) == str(other)


class Mappable: 

    def fmap(self, f):
        return self.__class__((f(x) for x in self))

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

    def __pow__(self, exp):
        return self.__class__(p for p in product(self, repeat=exp))
