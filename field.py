import torch

from dict import Dict
from hypergraph import Hypergraph

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

        def df (_, a): 
            faces = ((i, a.face(i)) for i in range(n + 2)) 
            df_i = (fmap(a, b)(self[b]) for i, b in faces)
            return sum(((-1)**i * df_i for i, b in faces))

        return self.complex.field(n + 1).map(df)
    
    def codiff(self): 
        n = self.degree 
        fmap = self.complex.extend
        cofaces = self.complex.cofaces(n - 1)

        def delta_f(_, b):
            delta_i = lambda i : \
                sum((fmap(a, b)(self[b]) for a in cofaces[i][b])
            return sum((delta_i(i) for i in range(len(cofaces))))

    def __add__(self, other):
        if type(other) in (int, float):
            return self.map(lambda vk, k: vk + other)
        return self.map(lambda vk, k: vk + other[k])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) in (int, float):
            return self.map(lambda vk, k: vk * other)
        return self.map(lambda vk, k: vk * other[k])
    
    def __rmul__(self, other):
        return self.__mul__(other)

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

    def __getitem__(self, face): 
        return self.values[face]

    def __repr__(self): 
        return f"{self.degree}-Field {self.values}"
