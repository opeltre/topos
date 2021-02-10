from itertools import product

from dict import Dict
from mixin import Hashable, Mappable
 

class SetMixin (Hashable, Mappable):
    """
    Hashable sets as strings.
    """
    def graph(self, f): 
        return Setmap({a: f(a) for a in self})

    def fibers(self, f):
        F = super().fibers(f)
        return Setmap(F).fmap(Set)

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


class Set (SetMixin, set):

    def __init__(self, elems=(), sep=':'):
        self.sep = sep 
        if type(elems) == str:
            elems = elems.split(sep) if len(elems) > 0 else []
        super().__init__(elems)
    
    def curry(self): 
        f = {}
        for (x, y) in self:
            if x in f:
                f[x].add(y)
            else:
                f[x] = set((y))
        return Setmap(f).fmap(Set)


class MapMixin (Mappable): 

    def __call__(self, arg):
        return self[arg]

    def fmap(self, f): 
        return self.__class__({k: f(v) for v, k in self})

    def map(self, f): 
        return self.__class__({k: f(v, k) for v, k in self})
    
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
    
    def domain(self): 
        return Set(x for fx, x in self)

    def __matmul__(self, other): 
        fog = {k: self[gk] for gk, k in other}
        return Setmap(fog)

    def __or__(self, other): 
        if isinstance(other, MapMixin):
            return other @ self
        else:
            return self.fmap(other)

    def __mul__(self, other): 
        if not isinstance(Mapmixin, other): 
            return Set(self) * other
        fg = {}
        for (fx, x) in self:
            for (gy, y) in other:
                fg[(x, y)] = (fx, gy)
        return Setmap(fg)

    def __add__(self, other): 
        f_g = {}
        for (gy, y) in other:
            f_g[y] = gy
        for (fx, x) in self: 
            f_g[x] = fx
        return self.__class__(f_g)
    
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
        return s

    def __repr__(self):
        s = str(self)
        b = '{\n' if '\n' in s else "{"  
        return b + str(self) + "}"


class Setmap (MapMixin, SetMixin, Dict):

    def __repr__(self):
        return f"Setmap {str(self)}" 

    def fibers(self, f=None): 
        return super().fibers(f).fmap(Set)
