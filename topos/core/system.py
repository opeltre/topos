from topos.base import Hypergraph, Chain, Cell, Shape

from .domain import Domain, GradedDomain, EmptyDomain
from .field import Field

from .operators import face, codifferential, zeta, invert_nil, nabla

from .functional import Functional, GradedFunctional
from .linear import Linear, GradedLinear

import torch

class System : 

    def __init__(self, K, shape=2, degree=-1, close=1, sort=1, void=1):

        #--- Closure for `cap` ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() if close else K
        if not void: 
            K = Hypergraph((r for r in K if len(r) > 0))
        self.hypergraph = K

        #--- Shapes of local tensors ---
        E = lambda i: shape if type(shape) == int else shape[i]
        self.shape = { 
            a: Shape(*(E(i) for i in a.list())) for a in K
        }

        #--- Nerve ---
        N = K.nerve(degree, sort=sort)
        self.degree = len(N) - 1
        shape = lambda chain : self.shape[chain[-1]]
        self.nerve = [GradedDomain(Nk, shape, k) for k, Nk in enumerate(N)]

        #--- Topological operators ---
        delta = [codifferential(self, i + 1) for i in range(self.degree)]
        d     = [dti.t() for dti in delta] 
        self.d      = GradedLinear([self], d + [0]      , 1 , "d")
        self.delta  = GradedLinear([self], [0] + delta  , -1, "d*")

        #--- Combinatorial operators ---
        zt = zeta(self, self.degree)
        mu = [invert_nil(zti, order=self.degree, tol=0) for zti in zt]
        self.zeta   = GradedLinear([self], zt, 0, "\u03b6")
        self.mu     = GradedLinear([self], mu, 0, "\u03bc")

        #--- Effective Energy gradient --- 
        d1 = face(self, 1, 1).t()
        d0 = face(self, 1, 0).t()
        def Deff (U): 
            return d0 @ U + torch.log(d1 @ torch.exp(- U))
        self.Deff = Functional.map([self[0], self[1]], Deff, "\u018a")

    def nabla(self, p, degree=0):
        """ Return the tangent map of Deff at p. """ 
        Ds = [nabla(self, d, p.data) for d in range(0, degree + 1)]
        return GradedLinear([self], Ds, 1, "\u2207_p")

    def __getitem__(self, degree):
        """ Return the domain instance at a given degree. """
        if degree < 0 or degree > self.degree:
            return EmptyDomain()
        return self.nerve[degree]

    def __repr__(self): 
        return "System [\n\n" \
            +  ",\n\n".join([str(Nk) for Nk in self.nerve]) \
            +  "\n\n]"
