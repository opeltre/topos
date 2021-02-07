import torch
import itertools 

from dict import Dict, Record
from hypergraph import Hypergraph
from simplex import Simplex

class Field (Tensor):

    def __init__(self, K, n=0, values={}):
        self.complex = K
        self.degree = n 
        self.cuda = False
        super().__init__(values)

    def map_cuda(self, f):
        out = Tensor()
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
