import fp
import torch

from topos.base import Nerve, Domain
from topos.core import Linear, linear_cache, face
from topos.core import Smooth, VectorField

def graded_map(method):
    def run(self, x, *args, **kwargs):
        if not isinstance(x, int) and 'degree' in dir(x):
            return method(self, x.degree, *args, **kwargs)(x)
        return method(self, x, *args, **kwargs)
    return run

def temperature_map(method):
    def run(self, H, beta=1):
        if isinstance(H, (float, int)):
            return method(self, H)
        return method(self, beta)(H)
    return run

class Network(Nerve):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def Point(cls):
        P = cls([[0]])
        P.__name__ = '.'
        return P

    @graded_map
    def exp_(self, d, beta=1):
        """
        Exponential map N[d] -> N[d] from energies to positive densities. 

        Returns `H -> exp(- beta * H)`. 
        """
        @Smooth(self[d], self[d])
        def f(t):
            return torch.exp(- beta * t.data)

        f.__name__ = "(e-)"
        return f

    @graded_map
    def _ln(self, d, beta=1):
        """
        Logarithmic map N[d] -> N[d] from positive densities to energies.

        Returns `G -> - ln(G) / beta`. 
        """
        @Smooth(self[d], self[d])
        def f(t): 
            return - torch.log(t.data) / beta

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
        @Smooth(self[d], self[d])
        def gibbs(H):
            y = self.exp_(d, beta)(H.data)
            Z = self.to_scalars(d)(y)
            return y / self.from_scalars(d)(Z)

        return gibbs

    def uniform(self, d=None):
        """
        Uniform beliefs on N[d].
        """
        return self.gibbs(self.zeros(d))

    @temperature_map
    def freeEnergy(self, beta=1):
        """
        Local free energies F: N[0] -> R[0].
        """
        e_  = self.exp_(0, beta)
        _ln = self.scalars()._ln(0, beta)
        sum = self.to_scalars(0)

        @Smooth(self[0], self.scalars()[0])
        def F(H):
            return _ln(sum(e_(H)))
        
        return F
    
    @temperature_map
    def freeDiff(self, beta=1):
        """
        Conditional free energy differences N[0] -> N[1].

        Given an input energy H in N[0], the free energy
        difference D(H) associates to every 1-chain 'a > b' in N[1]

            D(H)[a, b] = H[b] + ln sum_{a - b} exp(-H[a])

        The inverse temperature parameter `beta` acts as
        conjugation by energy scalings: 

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

        @Smooth(self[0], self[1])
        def D(H):
            return d0(H) - ln1 (d1(e0(H)))
        
        return D

    @linear_cache('face0')
    def face0(self):
        return super().face(0, 0)
    
    @linear_cache('face1')
    def face1(self):
        return super().face(0, 1)

    @temperature_map
    def freeBethe(self, beta=1):
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

        @Smooth(self[0], self.Point()[0])
        def F_Bethe(H):
            return (c * F(H)).data.sum()
        
        return F_Bethe

    @linear_cache("\u03b6\u03b4\u03bc")
    def codiff_zeta(self, d=1):
        return self.zeta(d - 1) @ self.codiff(d) @ self.mu(d)