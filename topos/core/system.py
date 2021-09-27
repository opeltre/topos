from topos import Hypergraph, Chain, Cell, Shape
from .field import Field
from .domain import GradedDomain

from .operators import differential, zeta, invert_nil
from .matrix import Matrix

import torch

class System : 
    
    def __init__(self, K, shape=2, close=True, sort=True, degree=-1):
       
        #--- Closure for `cap` ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() if close else K
        self.hypergraph = K

        #--- Nerve ---
        N = K.nerve(degree)
        if sort: 
            for Nk in N:
                Nk.sort(key = lambda c : (-len(c[-1]), str(c)))
        self.degree = len(N) - 1

        #--- Shapes of local tensors ---
        E = lambda i: shape if type(shape) == int else shape[i]
        self.shape = { 
            a: Shape(*(E(i) for i in a.list())) for a in K
        }

        #--- Graded Domains ---
        shape = lambda chain : self.shape[chain[-1]]
        self.nerve = [
            GradedDomain(self, k, Nk, shape) for k, Nk in enumerate(N)
        ]

        #--- Topological operators ---
        d = [differential(self, i) for i in range(self.degree)]
        delta = [di.t() for di in d][::-1] 
        self.d = [Matrix(di, 1, "d") for di in d] + [0]
        self.delta = [0] + [Matrix(dti, -1, "d*") for dti in delta]

        #--- Combinatorial operators ---
        zt = zeta(self, self.degree)
        mu = [invert_nil(zti, order=self.degree, tol=0) for zti in zt]
        self.zeta = [Matrix(zti, 0, "\u03b6") for zti in zt]
        self.mu = [Matrix(mui, 0, "\u03bc") for mui in mu]

    def __getitem__(self, degree):
        return self.nerve[degree]

    def __repr__(self): 
        return "System [\n\n" \
            +  ",\n\n".join([str(Nk) for Nk in self.nerve]) \
            +  "\n\n]"
