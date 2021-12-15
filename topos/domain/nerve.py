from .complex   import Complex

from topos.base import Hypergraph 
from topos.core import GradedLinear
from topos.core.operators import zeta, invert_nil

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
        T = self.scalars if not self.trivial else self
        self.bethe = T.mu[0].t() @ T.ones(0)

