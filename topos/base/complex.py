from topos.core import sparse
import topos.base.graph

import torch
from torch import cat, stack, arange

class Complex (topos.base.graph.Graph):
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

    def diff(self, d):
        """ Differential d: K[d] -> K[d + 1]. """
        src, tgt = self[d], self[d + 1]
        N, P = tgt.shape[0], src.shape[0]
        i = self.index(tgt) - self.index(tgt[0])
        out = sparse.matrix([N, P], [])
        for k in range(d + 2):
            j0  = self.index(src[0])
            j   = self.index(face(k, self[d + 1])) - j0
            val = (-1.) ** k
            out += sparse.matrix([N, P], stack([i, j]), val, t=0)
        return out
    
    @staticmethod
    def face(i, f):
        """ Face map acting on (batched) cells [j0, ..., jn]. """
        n = f.shape[-1]
        return f.index_select(f.dim() - 1, cat([arange(i), arange(i+1, n)]))
    
    @staticmethod 
    def simplices (faces):
        faces = parseTensor(faces)
        K = [[(0, faces)]]
        def nextf (nfaces):
            faces = []
            for i, f in nfaces:
                faces += [(j, cls.face(j, f)) for j in range(i, f.shape[-1])]
            return faces
        for codim in range(faces.shape[-1] - 1):
            K += [nextf(K[-1])]
        K_ = [cat([f for i, f in Kn[::-1]]) for Kn in K[::-1]]
        return K_

    @classmethod
    def simplicial(cls, faces):
        """ Simplicial closure. 
            
            Returns the complex containing every subface of the input faces.
        """
        return cls(*simplices(faces))

    def __repr__(self):
        return f'{self.dim} Complex {self}'

