from .domain    import Domain 

from topos.base import Shape, Fiber, Simplex, Hypergraph
from topos.core import Field, Functional, Linear, sparse
from topos.core.operators import from_scalar
from topos.core.sparse    import eye

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
    def free(cls, K, shape=None, degree=0):
        """
        Free sheaf over K with shapes F[i:j] = F[i] * F[j].
        """
        #--- Shapes of local tensors ---
        K = K if isinstance(K, Hypergraph) else Hypergraph(K)
        def getshape (region): 
            js = region.list()
            Es = [shape if type(shape) == int else shape[j] for j in js]
            return Shape(*Es)
        return cls(K, getshape if shape else None, degree)


    def __init__(self, keys, shape=None, degree=None, ftype=Fiber):
        """
        Create a sheaf from a dictionary of fiber shapes.
        """
        self.trivial = (shape == None) and not isinstance(keys, dict)

        super().__init__(keys, shape, degree, ftype)

        #--- Trivialise sheaf ---
        if 'scalars' not in self.__dir__():
            if self.trivial: 
                self.scalars = self
            elif isinstance(keys, dict):
                self.scalars = self.__class__(keys.keys(), degree=degree)
            else: 
                self.scalars = self.__class__(keys)
        
        #--- From/to scalars ---
        src, J = self.scalars, from_scalar(self)
        extend = Linear([src, self], J, "J")
        sums   = Linear([self, src], J.t(), "\u03a3")
        if not self.trivial:
            sizes  = extend @ sums @ self.ones()
            means  = (1/sizes) * sums

    
        #   =   =   Statistics  =   =   =

        #--- Energies / log-likelihoods ---
        _ln     = self.map(lambda d: -torch.log(d), "(-ln)")
        _lnT    = self.scalars.map(lambda d: -torch.log(d))
        #--- Gibbs states / densities ---
        exp_    = self.map(lambda d: torch.exp(-d), "(e-)")
        #--- Free energy ---
        freenrj = _lnT @ sums @ exp_
        #--- Normalisation ---
        norm    = Functional([self], lambda f: f/sums(f), "(1 / \u03a3)")
        gibbs   = (norm @ exp_).rename("(e- / \u03a3 e-)")
        
        #--- Local Fourier transforms
        N = self.size
        FT, iFT = {}, {}
        for a, Ea in self.items(): 
            if Ea.shape not in FT:   
                FT[Ea.shape]  = sparse.Fourier(Ea.shape)
                iFT[Ea.shape] = sparse.iFourier(Ea.shape)
        if N:
            FTs  = [(Ea, FT[Ea.shape])  for a, Ea in self.items()]
            iFTs = [(Ea, iFT[Ea.shape]) for a, Ea in self.items()]

            ij  = torch.cat([(Fa[0] + Ea.begin).t() for Ea, Fa in FTs])
            ji  = torch.cat([(Fa[0] + Ea.begin).t() for Ea, Fa in iFTs])
            Fij = torch.cat([Fa[1] for Ea, Fa in FTs])
            Fji = torch.cat([Fa[1] for Ea, Fa in iFTs])
       
            fft  = sparse.matrix([N, N], ij, Fij) 
            ifft = sparse.matrix([N, N], ji, Fji)
        else:
            fft  = sparse.matrix([0, 0], [])
            ifft = sparse.matrix([0, 0], [])
        
        self.maps = {
            "id"        : Linear([self], eye(self.size)),
            "_ln"       : _ln   ,   
            "exp_"      : exp_  ,
            "freenrj"   : freenrj,
            "normalise" : norm  ,
            "gibbs"     : gibbs      
        }
        if not self.trivial: 
            self.maps |= {
                "extend"    : extend,
                "sums"      : sums,
                "means"     : means,
                "fft"       : Linear([self], fft,  name="Fourier"),
                "ifft"      : Linear([self], ifft, name="Fourier*")
            }
        for k, fk in self.maps.items():
            setattr(self, k, fk)

    def __repr__(self):
        p1 = self.degree if self.degree != None else ''
        p2 = "Domain" if self.trivial else "Sheaf"
        return f"{p1} {p2} {self}"


#--- Simplical Fibers ---

class Simplicial (Sheaf) : 
 
    """ Sheaf with simplicial keys. """
    
    def __init__(self, keys, shape=None, degree=0, **kwargs):
        super().__init__(keys, shape, degree, ftype=Simplex, **kwargs)
