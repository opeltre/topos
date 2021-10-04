from .domain import Domain 

from topos.base import join_cells, Chain, Shape, Cell
from topos.core import Field, Functional, Linear
from topos.core.operators import from_scalar

import torch

class Sheaf (Domain) : 
    """
    Sheaves hold a dictionnary of `Fiber` objects.

    Each fiber is a pointer to an index range
    of size `n1 * ... * nd` for a shape `(n1, ..., nd)`.
    """

    def __init__(self, keys, shape=None, degree=0):
        """
        Create a domain with hashable keys and given shapes.

        Params: - keys  : [Hashable]
        ------  - shape : {keys : Shape} | keys -> Shape

        """
        self.trivial = (shape == None)
        self.scalars = self.__class__(keys) if not self.trivial else self

        #--- Join Cells ---
        if shape == None:
            shape = {k : Shape() for k in keys}
        elif callable(shape):
            shape = {k: shape(k) for k in keys}
        cells, size = join_cells(keys, shape)

        super().__init__(cells, degree, size=size)

        
        #--- From/To scalar fields ---
        src, J = self.scalars, from_scalar(self)
        extend = Linear([src, self], J, "J")
        sums   = Linear([self, src], J.t(), "\u03a3")
        #--- Normalisation ---
        norm    = Functional([self], lambda f: f/sums(f), "(1 / \u03a3)")
        #--- Energies / log-likelihoods ---
        _ln     = self.map(lambda d: -torch.log(d), "(-ln)")
        #--- Gibbs states / densities ---
        exp_    = self.map(lambda d: torch.exp(-d), "(e-)")
        gibbs   = (norm @ exp_).rename("(e- / \u03a3 e-)")
        
        self.maps = {
            "extend"    : extend,
            "sums"      : sums  ,
            "normalise" : norm  ,
            "_ln"       : _ln   ,   
            "exp_"      : exp_  ,
            "gibbs"     : gibbs      
        }
        for k, fk in self.maps.items():
            setattr(self, k, fk)

    def __repr__(self):
        return f"Sheaf  {self}" if not self.trivial else f"Domain {self}"

#--- System Fibers ---

class Simplicial (Sheaf) : 
    """ Domain with simplicial Chain keys """

    def __init__(self, keys, shape=None, degree=0):
        chains = [Chain.read(k) for k in keys]
        shape  = {Chain.read(k): shape(k) for k in keys} if shape\
                else None
        super().__init__(chains, shape, degree)
    
    def get(self, key):
        return super().get(Chain.read(key))

    def __repr__(self):
        return f"{self.degree} {super().__repr__()}"
