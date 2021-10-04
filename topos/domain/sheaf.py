from .domain import Domain 

from topos.base import Chain, Shape, Fiber, Hypergraph
from topos.core import Field, Functional, Linear
from topos.core.operators import from_scalar

import torch

class Sheaf (Domain) : 
    """
    Sheaves hold a dictionnary of Fiber objects.

    Each fiber is a pointer to an index range
    of size `n1 * ... * nd` for a shape `(n1, ..., nd)`.

    See Sheaf.range() for a visual representation of indices.
    """

    @classmethod
    def closure(cls, hypergraph, shape=None, degree=0, void=1):
        K = hypergraph if isinstance(hypergraph, Hypergraph)\
            else Hypergraph(hypergraph)
        K = K.closure()
        if not void:
            K = Hypergraph((r for r in K if len(r)))
        return cls.free(K, shape, degree)
            
    
    @classmethod
    def free(cls, hypergraph, shape=None, degree=0):
        #--- Shapes of local tensors ---
        K = hypergraph if isinstance(hypergraph, Hypergraph)\
            else Hypergraph(hypergraph)
        def getshape (region): 
            js = region.list()
            Es = [shape if type(shape) == int else shape[j] for j in js]
            return Shape(*Es)
        return cls(K, getshape if shape else None, degree)


    def __init__(self, keys, shape=None, degree=0):
        """
        Create a sheaf from a dictionary of fiber shapes.
        """
        self.trivial = (shape == None)
        self.scalars = self.__class__(keys) if not self.trivial else self

        #--- Join Fibers ---
        if shape == None:
            shape = {k : Shape() for k in keys}
        elif callable(shape):
            shape = {k: shape(k) for k in keys}
        fibers, size = Fiber.join(keys, shape)

        super().__init__(fibers, degree, size=size)

        
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
