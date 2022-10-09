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
        Exponential map N[d] -> N[d] from energies to positive densities. 

        Returns `H -> exp(- beta * H)`. 
        """
        src = self.Field(d)
        f = fp.Arrow(src, src)(lambda t: torch.exp(- beta * t.data))
        f.__name__ = "(e-)"
        return f

    @graded_map
    def _ln(self, d, beta=1):
        """
        Logarithmic map N[d] -> N[d] from positive densities to energies.

        Returns `G -> - ln(G) / beta`. 
        """
        src = self.Field(d)
        f = fp.Arrow(src, src)(lambda t: - torch.log(t.data) / beta)
        f.__name__ = "(-ln)"
        return f

    @graded_map
    def gibbs(self, d, beta=1):
        """
        Gibbs state map N[d] -> N[d] from energies to probabilities. 

        Returns `H -> exp(- beta * H) / Z` 
        
        These beliefs might be called 'softmins' of H 
        at inverse temperature beta = 1 / T. 
        They concentrate on minima of H when T goes to 0, 
        i.e. beta goes to infinity.
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
        Uniform beliefs on N[d].
        """
        return self.gibbs(self.zeros(d))

    def BetheFreeEnergy(self, beta=1):
        """
        Bethe free energy F_Bethe: N[0] -> R. 

        The Bethe free energy of H is a weighted sum of 
        local free energies 

            F_Bethe(H) = sum [c[a] * F(H)[a] for a in N[0]]

        The Bethe-Kikuchi coefficients satisfy for all b in N[0]
        the inclusion-exclusion principle:

            sum [c[a] for a >= b] == 1

        Hence F_Bethe(H) provides a combinatorially reasonable 
        local approximation of the global free energy. 
        """
        F = self.freeEnergy(beta)
        c = self.bethe(0)

        @fp.Arrow(self.Field(0), fp.Tens([1]))
        def F_Bethe(H):
            return (c * F(H)).data.sum()
        
        return F_Bethe


    def freeEnergy(self, beta=1):
        """
        Local free energies F: N[0] -> R[0].
        """
        e_  = self.exp_(0, beta)
        _ln = self.scalars()._ln(0, beta)
        sum = self.to_scalars(0)

        @fp.Arrow(self.Field(0), self.scalars().Field(0))
        def F(H):
            return _ln(sum(e_(H)))
        
        return F
    
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