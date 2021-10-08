from topos.base import Hypergraph, Chain, Fiber, Shape

from .domain    import Domain
from .cartesian import Empty
from .sheaf     import Simplicial

from topos.core import\
     Field, Linear, GradedLinear, Functional, GradedFunctional 

from topos.core.operators import\
    face, codifferential, zeta, invert_nil, nabla

import torch


class Complex (Domain): 
    
    def __init__(self, domains):
        self.grades = domains 
        self.degree = len(domains) - 1
        self.trivial = domains[0].trivial
        if "scalars" not in self.__dir__():
            self.scalars = Complex([di.scalars for di in domains])\
                           if not self.trivial else self

        #--- Topological operators ---
        delta = [codifferential(self, i + 1) for i in range(self.degree)]
        d     = [dti.t() for dti in delta]
        self.d      = GradedLinear([self], d + [0]      , 1 , "d")
        self.delta  = GradedLinear([self], [0] + delta  , -1, "d*")
        
        #--- Lift functorial maps to all degrees --- 
        self.lifts = {}
        for k, fk in self.grades[0].maps.items():
            src = self.scalars if k == "sums"   else self
            tgt = self.scalars if k == "extend" else self
            self.lifts[k] = self.lift(k, fk.name, src, tgt)
        for k, lfk in self.lifts.items():
            setattr(self, k, lfk)

    def get(self, key): 
        """ Retrieve a domain cell from its key. """
        if not isinstance(key, Chain):
            key = Chain.read(key)
        return self[key.degree][Chain.read(key)]

    def lift (self, f, name="name", src=None, tgt=None):
        src = src if src else self
        tgt = tgt if tgt else tgt
        fs = [Ni.__getattribute__(f) for Ni in self.grades] 
        return GradedLinear([src, tgt], fs, 0, name)\
            if isinstance(fs[0], Linear)\
            else GradedFunctional([src, tgt], fs, 0, name=name)

    def __getitem__(self, degree):
        """ Return the domain instance at a given degree. """
        if degree < 0 or degree > self.degree:
            return Empty(degree)
        return self.grades[degree]

    def __iter__(self):
        for Kd in self.grades:
            for cell in Kd:
                yield cell

    def field (self, data=None, degree=0):
        return self[degree].field(data)

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
    
    @classmethod
    def closure(cls, K, shape=2, degree=-1, void=1):
        """ Closure for `cap`. """
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() 
        if not void: 
            K = Hypergraph((r for r in K if len(r) > 0))
        return cls(K, shape, degree)


    def __init__(self, K, shape=None, degree=-1, sort=1):

        #--- Scalars ---
        self.trivial = (shape == None)
        self.scalars = self.__class__(K, None, degree) \
                       if not self.trivial else self
                       # /!\ do not recompute the nerve /!\

        #--- Nerve ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        self.hypergraph = K
        N = K.nerve(degree, sort=sort)
        E = shape if shape else None
        self.nerve  = [Simplicial(Nk, E, degree=k)\
                       for k, Nk in enumerate(N)]
        self.degree = len(N) - 1

        super().__init__(self.nerve)

        #--- Combinatorial operators ---
        zt = zeta(self, self.degree)
        mu = [invert_nil(zti, order=self.degree, tol=0) for zti in zt]
        self.zeta   = GradedLinear([self], zt, 0, "\u03b6")
        self.mu     = GradedLinear([self], mu, 0, "\u03bc")


class System (Nerve): 
    """ 
    Simplicial nerve N[d](K, E) of a free sheaf E over K.


    The local shape of region `a` in N[0](K, E) is given by:

        E[a] = (E[i] for i in a)

    The shape of a chain `a0 > ... > ad` in N[d](K, E) is then:
        
        E[a0 > ... > ad] = E[ad]

    """
    def __init__(self, K, shape=2, degree=-1, sort=1):
        """ System on the nerve of hypergraph K. """

        #--- Shapes of local tensors ---
        def getshape (chain): 
            js = chain[-1].list()
            Es = [shape if type(shape) == int else shape[j] for j in js]
            return Shape(*Es)

        super().__init__(K, getshape if shape else None, degree)
       
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
    
    def __repr__(self): 
        return f"System {self}"
