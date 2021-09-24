from topos import Hypergraph, Chain, Cell, Shape
from .field import Field
from .domain import Domain_k

import torch

class System : 
    
    def __init__(self, K, shape=2, close=True, sort=True, degree=-1):
        
        #--- Nerve ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() if close else K
        N = K.nerve(degree)
        if sort: 
            for Nk in N:
                Nk.sort(key = lambda c : (-len(c[-1]), str(c)))
        self.degree = len(N)

        #--- Shapes of local tensors ---
        E = lambda i: shape if type(shape) == int else shape[i]
        self.shape = { 
            a: Shape(*(E(i) for i in a.list())) for a in K
        }

        #--- Pointers to start of local data ---
        shape = lambda chain : self.shape[chain[-1]]
        self.nerve = [
            Domain_k(self, k, Nk, shape) for k, Nk in enumerate(N)
        ]
    
    def __getitem__(self, degree):
        return self.nerve[degree]

    def __repr__(self): 
        return "System [\n\n" \
            +  ",\n\n".join([str(Nk) for Nk in self.nerve]) \
            +  "\n\n]"
