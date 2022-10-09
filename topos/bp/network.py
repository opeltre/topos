import fp
import torch

from topos.base import Nerve
from topos.core import linear_cache, face


def graded_map(method):
    def run(self, x, *args, **kwargs):
        if isinstance(x, fp.Tensor):
            return method(self, x.degree, *args, **kwargs)(x)
        return method(self, x, *args, **kwargs)
    return run


class Network(Nerve):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @graded_map
    def exp_(self, d, beta=1):
        """
        Exponential map from energies to unnormalized Gibbs densities. 

        Returns `H -> torch.exp(- beta * H)`. 
        """
        src = self.Field(d)
        f = fp.Arrow(src, src)(lambda t: torch.exp(- beta * t.data))
        f.__name__ = "(e-)"
        return f

    @graded_map
    def _ln(self, d, beta=1):
        """
        Logarithmic map from densities to energies.

        Returns `G -> - torch.log(G) / beta`. 
        """
        src = self.Field(d)
        f = fp.Arrow(src, src)(lambda t: - torch.log(t.data) / beta)
        f.__name__ = "(-ln)"
        return f

    @graded_map
    def gibbs(self, d, beta=1):
        """
        Normalized Gibbs density. 

        Defined by `p(H) = exp(- beta * H) / Z` 
        
        These beliefs may sometimes be called 'softmins' of H at inverse
        temperature beta.
        """
        src = self.Field(d)
        def p(H):
            y = self.exp_(d, beta)(H.data)
            Z = self.to_scalars(y)
            return y / self.from_scalars(Z)
        f = fp.Arrow(src, src)(p)
        f.__name__ = "gibbs"
        return f

    def uniform(self, d=None):
        """
        Uniform beliefs.
        """
        return self.gibbs(self.zeros(d))
        
    def freeEnergy(self, x):
        pass
    
    def freeGrad(self, beta=1):
        """
        Free Energy Gradient N[0] -> N[1].

        Given an input energy H in N[0], the free energy
        gradient D(H) in N[1] associates to every 1-chain 'a > b'
        the difference:

            D(H)[a, b] = H[b] + ln sum_{a - b} exp(-H[a])

        The inverse temperature parameter `beta` represents conjugation by energy 
        scalings: 

            D_beta(H) = D(beta * H) / beta
    
        Higher beta = 1 / T means lower temperature and higher non-linearity. 

        The tangent map at H of D is a linear operator N[0] -> N[1] 
        where conditional free energies are replaced by conditional expectations
        with respect to local Gibbs states induced by H. 

        The tangent map TD(H) extends to a differential if on N[.] iff
        local gibbs states are consistent.
        """

        d0, d1 = self.face0(), self.face1()
        e0, ln1 = self.exp_(0, beta), self._ln(1, beta)

        @fp.Arrow(self.Field(0), self.Field(1))
        def D(H):
            return d0(H) - ln1 (d1(e0(H)))
        
        return D



    @linear_cache('face0')
    def face0(self):
        return super().face(0, 0)
    
    @linear_cache('face1')
    def face1(self):
        return super().face(0, 1)


def GBPDiffusion(network):
    pass

def BetheDiffusion(network):
    pass