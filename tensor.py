import torch 
from functools import reduce

from set import MapMixin
from dict import Record 

class VectorMixin(MapMixin): 

    def __add__(self, other):
        if type(other) in (int, float):
            return self.map(lambda ti, i: ti + other)
        D = self.domain() + other.domain()
        s = lambda i:\
            (self[i] if i in self else 0) +\
            (other[i] if i in other else 0)
        return self.__class__({i: s(i) for i in D}).trim()

    def __sub__(self, other): 
        return self.__add__(-1 * other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) in (int, float):
            return self.map(lambda tj, j: tj * other)
        s = self.map(lambda tj, j: other[j] if j in other else 0)
        return self.map(lambda tj, j: tj * s[j]).trim()
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self): 
        s = super().__repr__()
        return f"Tensor {s}"

    def __and__(self, other):
        return (self * other).sum()

    def sum(self):
        isTensor = lambda t:\
            isinstance(t, (torch.Tensor, VectorMixin))
        return sum(
            ti.sum() if isTensor(ti) else ti \
            for ti, i in self)

    def trim(self): 
        cls = self.__class__
        not_zero = lambda ti, i: not cls.iszero(ti)
        trims = lambda t: t.trim() if isinstance(t, VectorMixin)\
                else t
        return cls(self.fmap(trims).filter(not_zero))

    @classmethod 
    def iszero(cls, t): 
        if not isinstance(t, VectorMixin): 
            return t == 0
        return len(t.filter(lambda ti, i: not cls.iszero(ti))) == 0


class Product (VectorMixin, tuple):

    def __iter__(self): 
        return ((xi, i) for i, xi in enumerate(super().__iter__()))
    
    def __str__(self):
        s = ""
        for xi, i in self:
            si = str(xi)
            if '\n' in si: 
                si = ('\n' + si).replace('\n', '\n   ') 
                s += f"\n{i} :{si}"
            else:
                s += f" . {si}" if len(s) > 0 else si
        return s

    def __repr__(self): 
        s = super().__repr__()
        return f"\u03a0-{s}"

    def __or__(self, other): 
        return self.__class__(xi for xi, i in (*self, *other))

    def project(self, dim=(0,)):
        return self.restrict(dim)

    def p(self, *args):
        return self.project(*args)

    def flip(self, dim=(1, 0)):
        k = len(dim)
        return self.__class__(
            (self[dim[i]] if i < k else xi for xi, i in self))

    def fmap(self, f):
        return self.__class__((f(xi) for xi, i in self))

    def map(self, f):
        return self.__class__((f(xi, i) for xi, i in self))


class Tensor (VectorMixin, Record):  
    
    def __init__(self, d={}): 
        self.degree = None
        tensor = lambda t: t \
            if isinstance(t, (VectorMixin, int, float, torch.Tensor)) \
            else Tensor(t)
        word = lambda a: a if isinstance(a, Product) else Product(a) 
        super().__init__({
            word(a): tensor(da) for da, a in Record(d)
        })

    def __or__(self, other): 
        t = {}
        otimes = lambda fa, gb: fa | gb \
                if isinstance(fa, Tensor) and isinstance(gb, Tensor)\
                else fa * gb 
        for fa, a in self:
            for gb, b in other:
                t[a | b] = otimes(fa, gb)
        return self.__class__(t)
        
    def sum(self, dim=None):
        if dim == None:
            return super().sum()
        proj = lambda ta, a: a.forget(dim)
        return self.fibers(proj).fmap(lambda tb: tb.sum())

    def curry(self, dim=0):
        proj = lambda ta, a: a.restrict(dim)
        t = self.fibers(proj).fmap(lambda tb: tb.sum(dim))
        return Matrix(t)

    def flip(self, dim=(1,0)):
        t = lambda a: a.flip(dim)
        return self.__class__({
            t(a): fa for fa, a in self
        })

    def cup(self, other):
        r = self.__class__() 
        s = other.fibers(lambda sb, b: b.p(0))
        cup = lambda ti, sj: ti.cup(sj) if\
            isinstance(ti, Tensor) and isinstance(sj, Tensor) \
            else ti * sj
        for ta, a in self: 
            a_ = a.forget(-1)
            j = a .p(-1)
            for sb, b in s[j] if j in s else []:
                r[a_ | b] = cup(ta, sb)
        return r.trim()

    def cap(self, other): 
        r = self.__class__()
        cap = lambda ti, sj: ti.cap(sj) if\
            isinstance(ti, Tensor) and isinstance(sj, Tensor) \
            else ti * sj
        k_1 = 0
        sk = Tensor()
        for tb, b in self: 
            k = len(b)
            sk = other.curry(range(0, k)) if k_1 != k else sk
            k_1 = k
            for sa, a in sk[b]:
                c = a.forget(range(0, k - 1))
                r[c] = cap(tb, sa) + (r[c] if c in r else 0)
        return r.trim()

    def cup_pows(self, other=None, N=10):
        p = other if other else self
        pk, pows, i = self, [], 0
        while not Tensor.iszero(pk) and i <= N:
            pows += [pk] 
            pk = pk.cup(p)
            i += 1
        return Product(pows)

class Vector (Tensor): 
    # __init__ override: Vector({a: 1}) -> Tensor({(a,): 1})
    # or
    # def Vector(v) 
    pass


class Matrix (Tensor): 

    def uncurry(self): 
        t = {}
        for ta, a in self:
            for tab, b in ta:
                t[a | b] = tab
        return Tensor(t)

    def __matmul__(self, other): 
        if isinstance(other, Matrix):
            B = other.t()
            AB = self.fmap(
                lambda Ai_: B.fmap(
                lambda B_k: Ai_ & B_k))
            return AB.trim()
        return Tensor(A).map(lambda Ai, i: Ai & other[i])

    def transpose(self):
        return self.t()

    def t(self):
        T = Tensor()
        for Ai, i in self:
            for Aij, j in Ai:
                T += Tensor({j: {i: Aij}})
        return self.__class__(T)
