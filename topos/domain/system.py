from topos.base import Hypergraph, Chain, Fiber, Shape

from .domain    import Domain
from .cartesian import Empty, Sum
from .sheaf     import Simplicial

from topos.core import\
     Field, Linear, GradedLinear, Functional, GradedFunctional 

from topos.core.operators import\
     coface, codifferential, zeta, invert_nil, nabla

import torch


class Complex (Sum): 
    """
    Simplicial complexes C[i](K, E) over K with values in E. 

    Topological operators d and d* satisfying the (co)differential rule!
        
        d @ d = 0   and     d* @ d* = 0

    act on C[.](K, E) with degrees +1 and -1 respectively
    [See C.d, C.delta]
    """
    def __init__(self, *domains):
        super().__init__(*domains)

        #--- Topological operators ---
        delta = [codifferential(self, i + 1) for i in range(self.rank)]
        d     = [dti.t() for dti in delta]
        self.d      = GradedLinear([self], d + [0]      , 1 , "d")
        self.delta  = GradedLinear([self], [0] + delta  , -1, "d*")
        
        #--- Lift functorial maps to all degrees --- 
        self.lifts = {}
        for k, fk in self.grades[0].maps.items():
            src = "scalars" if k == "sums"   else None
            tgt = "scalars" if k == "extend" else None
            self.lifts[k] = self.lift(k, fk.name, src, tgt)
        for k, lfk in self.lifts.items():
            setattr(self, k, lfk)

    def get(self, key): 
        """ Retrieve a domain cell from its key. """
        if not isinstance(key, Chain):
            key = Chain.read(key)
        return self[key.degree][Chain.read(key)]


    def __getitem__(self, degree):
        """ Return the domain instance at a given degree. """
        if degree < 0 or degree > self.rank:
            return Empty(degree)
        return self.grades[degree]

    def __str__(self): 
        return "[\n\n  " \
            +  ",\n\n  ".join([repr(Nk) for Nk in self.grades]) \
            +  "\n\n]"

    def __repr__(self):
        return f"Complex {self}"


class Nerve (Complex): 
    """
    Simplicial nerve N[d](K, E) of a sheaf E over a covering K.

    Combinatorial operations on the partial order of K = N[0]
    are extended to higher degrees [see K.zeta and K.mu].
    """
    def __init__(self, *nerve):
        K = nerve[0].fibers.keys()
        self.hypergraph = Hypergraph((a[-1] for a in K))
        super().__init__(*nerve)

        #--- Combinatorial operators ---
        zt = zeta(self, self.rank)
        mu = [invert_nil(zti, order=self.rank, tol=0) for zti in zt]
        self.zeta   = GradedLinear([self], zt, 0, "\u03b6")
        self.mu     = GradedLinear([self], mu, 0, "\u03bc")
        
        #--- Bethe numbers c[b] ---
        T = self.scalars
        self.bethe = T.mu[0].t() @ T.ones(0)

class System (Nerve): 
    """ 
    Simplicial nerve N[d](K, E) of a free sheaf E over K.


    The local shape of region `a` in N[0](K, E) is given by:

        E[a] = (E[i] for i in a)

    The shape of a chain `a0 > ... > ad` in N[d](K, E) is then:
        
        E[a0 > ... > ad] = E[ad]

    """
    @classmethod
    def closure(cls, K, shape=2, degree=-1, void=1, free=True):
        """ Closure for `cap`. """
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() 
        if not void: 
            K = Hypergraph((r for r in K if len(r) > 0))
        return cls(K, shape, degree, free)

    def __init__(self, K, shape=2, degree=-1, sort=1, free=True):
        """ System on the nerve of hypergraph K. """

        #--- Compute Nerve ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        self.hypergraph = K
        nerve = K.nerve(degree, sort=sort)
        self.rank = len(nerve) - 1

        #--- Scalar Fields ---
        self.trivial = (shape == None)
        if not self.trivial:
            N  = [Simplicial(Nk, None, degree=k)\
                           for k, Nk in enumerate(nerve)]
            self.scalars = Nerve(*N)

        #--- Tensor valued Fields --- 
        if type(shape) == int:
            E = lambda c : Shape(*[shape for j in c[-1].list()])
        elif free and callable(shape):
            E = lambda c : Shape(*[shape(j) for j in c[-1].list()])
        elif free and isinstance(shape, dict):
            E = lambda c : Shape(*[shape[j] for j in c[-1].list()])
        else:
            E = shape
        NE = [Simplicial(Nk, E, degree=k)\
                       for k, Nk in enumerate(nerve)]
        super().__init__(*NE)
       
        #--- Effective Energy gradient --- 
        if self.rank >= 1:
            d0 = coface(self, 0, 0)
            d1 = coface(self, 0, 1)
            def Deff (U): 
                return d0 @ U + torch.log(d1 @ torch.exp(- U))
            self.Deff = Functional.map([self[0], self[1]], Deff, "\u018a")
    
    def restriction(self, K): 
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        shape = {fiber.key[1] : fiber.shape for fiber in self}
        return self.__class__(K, shape, degree=self.rank, free=False)

    def nabla(self, p, degree=0):
        """ Return the tangent map of Deff at p. """ 
        Ds = [nabla(self, d, p.data) for d in range(0, degree + 1)]
        return GradedLinear([self], Ds, 1, "\u2207_p")
    
    def __repr__(self): 
        return f"System {self}"
