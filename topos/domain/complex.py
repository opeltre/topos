from .cartesian import Empty, Sum

from topos.base import Chain, Seq
from topos.core import GradedLinear

from topos.core.operators import coface, codifferential

import torch


class Complex (Sum): 
    """
    Simplicial complexes C[i](K, E) over K with values in E. 

    Topological operators d and d* satisfying the (co)differential rule:
        
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
        if isinstance(key, Seq):
            return self[key[0]][Chain.read(key[1])]
        if not isinstance(key, Chain):
            chain = Chain.read(key)
        return self[chain.degree][chain]

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


