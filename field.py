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

    def __repr__(self): 
        return f"{self.degree}-Field\n {self.values}"
