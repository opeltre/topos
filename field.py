import torch
import itertools 

from dict import Dict
from hypergraph import Hypergraph
from simplex import Simplex

class Field (Tensor):

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
