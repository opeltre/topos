from itertools import product, chain

from .hashable import Hashable

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


class Mapping: 

    # __init__({k : fk}) very inefficient! 2 dict creations. 

    def fmap(self, f):
        return self.__class__({k: f(x) for x, k in self})

    def map(self, f):
        return self.__class__({k: f(x, k) for x, k in self})

    def filter(self, f): 
        return self.__class__({k: x for x, k in self if f(x, k)})

    def domain(self): 
        return Set(x for fx, x in self)

    def fibers(self, f=None):
        F = {}
        if f == None:
            for y, x in self: 
                F[y] = F[y] + [x] if y in F else [x]
            return self.__class__(F) 
        for y, x in self: 
            z = f(y, x)
            if z in F:
                F[z][x] = y
            else:
                F[z] = {x: y} 
        return self.__class__(F).fmap(self.__class__)

    def uncurry(self): 
        g = {}
        for fx, x in self:
            for fxy, y in fx: 
                g[(x, y)] = fxy
        return self.__class__(g)

    def curry(self): 
        f = {}
        for gxy, (x, y) in self:
            if x in f:
                f[x][y] = gxy
            else:
                f[x] = {y: gxy}
        return self.__class__(f).fmap(self.__class__)
    
    def restrict(self, dim):
        dim = (dim, ) if type(dim) == int else dim
        return self.__class__((self[i] for i in dim))

    def forget(self, dim):
        dim = (dim, ) if type(dim) == int else dim
        n = len(self)
        dim = [i % n for i in dim]
        axes = [j for j in range(n) if j not in dim]
        return self.restrict(axes)

    def __str__(self):
        elems = [(str(k), str(ek)) for ek, k in self]
        elems.sort()
        s = ''
        for k, ek in elems: 
            sk = str(ek)
            if '\n' in sk: 
                sk = '\n' + sk 
            if '\n\n' in sk:
                sk += '\n'
            sk = sk.replace('\n', '\n\t')
            s += f"{str(k)} :-> {sk}\n"

        b = '{\n' if '\n' in s else "{"  
        return b + s + "}"

    def __repr__(self):
        return f"Mapping {str(self)}"


class Map (Mapping, Hashable, dict):
   
    def __repr__(self):
        return f"Map {str(self)}" 

    def __iter__(self): 
        return ((self[k], k) for k in super().__iter__())
