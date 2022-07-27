from topos.core import sparse, face, simplices, Linear, linear_cache
from topos.io   import readTensor
from .graph     import Graph

import torch
from torch import cat, stack, arange

class Complex (Graph):
    """
    Simplicial complexes.

    A simplicial complex is a hypergraph containing all the subsets of its cells.
    The i-th face map consists of removing the i-th vertex from a degree-n cell,
    for 0 <= i <= n.
    Alternated sums of face maps induce a dual pair of differential and codifferential
    (or boundary) operators, `d` and `delta = d*`.

    References:
    -----------
    - Peltre, 2020:
        Message-Passing Algorithms and Homology, Chapter II
        https://arxiv.org/abs/2009.11631
    """
    
    @linear_cache("d")
    def diff(self, d):
        """ Differential d: K[d] -> K[d + 1]. """
        src, tgt = self[d].keys, self[d + 1].keys
        N, P   = self[d+1].size, self[d].size
        N0, P0 = self.begin[d+1], self.begin[d]
        out = sparse.matrix([N, P], [])
        off = torch.stack([N0, P0]).long()
        for k in range(d + 2):
            Fk = self.fmap((tgt, face(k, tgt))).T - off 
            out += (-1.) ** k * sparse.matrix([N, P], Fk)
        return Linear(self[d], self[d + 1])(out, degree=1, name="d")
            
    
    @classmethod
    def simplicial(cls, faces):
        """ Simplicial closure.

            Returns the complex containing every subface of the input faces.
        """
        N = max(len(f) for f in faces)
        src = [[] * N]
        K = [F.reshape([-1, F.shape[-1]]) for F in simplices(faces)]
        return cls(K)

    def __repr__(self):
        return f'{self.dim} Complex {self}'
