import fp
import torch

from topos.base import Nerve

class Network(Nerve):
    
    def exp_(self, x, beta=1):
        """
        Exponential map from energies to unnormalized Gibbs densities. 

        Returns `torch.exp(- beta * x)`. 
        """
        if isinstance(x, int):
            src = self.Field(x)
            f = fp.Arrow(src, src)(lambda t: torch.exp(- beta * t.data))
            f.__name__ = "(e-)"
            return f
        elif isinstance(x, fp.Tensor):
            src = self.Field(x.degree)
            return src(torch.exp(-beta * x.data))

    def _ln(self, x, beta=1):
        """
        Logarithmic map from densities to energies.

        Returns `- torch.log(x) / beta`. 
        """
        if isinstance(x, int):
            src = self.Field(x)
            f = fp.Arrow(src, src)(lambda t: - torch.log(t.data) / beta)
            f.__name__ = "(-ln)"
            return f
        elif isinstance(x, fp.Tensor):
            src = self.Field(x.degree)
            return src(- torch.log(x.data) / beta)

    def gibbs(self, x, beta=1):
        """
        Normalized Gibbs density. 

        Defined by `p(x) = exp(- beta * x) / Z` (sometimes called a softmin).
        """
        if isinstance(x, int):
            src = self.Field(x)
            def p(H):
                y = self.exp_(x, beta)(H)
                Z = self.to_scalars(y)
                return y / self.from_scalars(Z)
            f = fp.Arrow(src, src)(p)
            f.__name__ = "gibbs"
            return f
        elif isinstance(x, fp.Tensor) and 'degree' in dir(x):
            return self.gibbs(x.degree, beta)(x)
        
    def freeEnergy(self, x):
        pass

    def freeGrad(self, x):
        # => cache face(0) and face(1) linear maps 
        pass


def GBPDiffusion(network):
    pass

def BetheDiffusion(network):
    pass