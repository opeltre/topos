from itertools import product, chain


class Iterable: 

    """ .__iter__ """

    def fmap(self, f):
        return self.__class__((f(x) for x in self))

    def foldl(self, f, acc=None):
        for x in self:
            acc = f(acc, x) 
                if not isinstance(acc, NoneType)
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


class Sortable (Iterable):

    """ .__contains__ """
    
    def __and__(self, other):
        return self.__class__(x for x in self if x in other)

    def __sub__(self, other): 
        return self.__class__(x for x in self if x not in other)

    def __or__(self, other):
        return self + (other - self)

    def __str__(self): 
        elems = [str(e) for e in self]
        elems.sort()
        return "(" + self.sep.join(elems) + ")"


class Set (Iterable, set):

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


