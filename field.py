import torch
import itertools 


from dict import Dict
from hypergraph import Hypergraph
from simplex import Simplex

class Field ():

    def __init__(self, K, n=0, values="zeros"):
        """
        """
        self.complex = K
        self.degree = n 
        self.cuda = False

        if type(values) == str: 
            if values in K.__dir__():
                val = K.__getattribute__(values)
                values = {a: val(a) for a in K[n]}
            else:
                values = {a: 0 for a in K[n]}
        self.values = Dict(values)

    def diff (self):
        n = self.degree
        fmap = self.complex.project
        faces = lambda a: enumerate(a.faces())
        dfi = lambda a, b: fmap(b[-1], a[-1])(self[b])
        df = lambda _, a: \
            sum(((-1)**i * dfi(a, b) for i, b in faces(a)))
        return self.field(n + 1).map(df)
    
    def codiff(self): 
        n = self.degree 
        fmap = self.complex.extend
        cofaces = self.complex.cofaces(n - 1)
        dfi = lambda b, i: \
            sum((fmap(a[-1], b[-1])(self[a]) for a in cofaces[i][b]))
        df = lambda _, b: \
            sum(((-1)**i * dfi(b, i) for i in range(len(cofaces))))
        return self.field(n - 1).map(df)

    def zeta (self):
        n = self.degree
        fmap = self.complex.extend 
        product = itertools.product
        icones = lambda a: (
            *(self.complex.intercone(a[i], a[i+1]) for i in range(n)),
            self.complex.below(a[n]))
        zfaces = lambda a: (Simplex(b) for b in product(*icones(a)))
        zf = lambda _, a: \
            sum(fmap(a[-1], b[-1])(self[b]) for b in zfaces(a))
        return self.field(n).map(zf)

    def __matmul__(self, other): 
        da, db = self.degree, other.degree

    def map(self, f): 
        values = self.values.map(f)
        return Field(self.complex, self.degree, values)

    def fmap(self, f):
        values = self.values.fmap(f)
        return Field(self.complex, self.degree, values)

    def map_cuda(self, f):
        out = Field({})
        for vk, k in self:
            sk = torch.cuda.Stream()
            with torch.cuda.stream(sk):
                out[k] = f(self[k], k)
        torch.cuda.synchronise()
        return out

    def field(self, degree): 
        return Field(self.complex, degree)

    def __add__(self, other):
        if type(other) in (int, float):
            return self.map(lambda vk, k: vk + other)
        return self.map(lambda vk, k: vk + other[k])

    def __sub__(self, other): 
        return self.__add__(-1 * other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) in (int, float):
            return self.map(lambda vk, k: vk * other)
        return self.map(lambda vk, k: vk * other[k])
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, face):
        if type(face) == tuple: 
            return self[face[0]][face[1:]]
        elif face == slice(None):
            return self
        if type(face) == str:
            face = Simplex(face)
        return self.values[face]

    def __iter__(self): 
        return self.values.__iter__()

    def __repr__(self): 
        return f"{self.degree}-Field {self.values}"
