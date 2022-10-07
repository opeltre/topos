import fp
import torch

from topos.base import Nerve


def graded_map(method):
    def run(self, x, *args, **kwargs):
        if isinstance(x, fp.Tensor):
            return method(self, x.degree, *args, **kwargs)(x)
        return method(self, x, *args, **kwargs)
    return run

class Network(Nerve):
    
    @graded_map
    def exp_(self, d, beta=1):
        """
        Exponential map from energies to unnormalized Gibbs densities. 

        Returns `torch.exp(- beta * x)`. 
        """
        src = self.Field(d)
        f = fp.Arrow(src, src)(lambda t: torch.exp(- beta * t.data))
        f.__name__ = "(e-)"
        return f

    @graded_map
    def _ln(self, d, beta=1):
        """
        Logarithmic map from densities to energies.

        Returns `- torch.log(x) / beta`. 
        """
        src = self.Field(d)
        f = fp.Arrow(src, src)(lambda t: - torch.log(t.data) / beta)
        f.__name__ = "(-ln)"
        return f

    @graded_map
    def gibbs(self, d, beta=1):
        """
        Normalized Gibbs density. 

        Defined by `p(x) = exp(- beta * x) / Z` (sometimes called a softmin).
        """
        src = self.Field(d)
        def p(H):
            y = self.exp_(d, beta)(H.data)
            Z = self.to_scalars(y)
            return y / self.from_scalars(Z)
        f = fp.Arrow(src, src)(p)
        f.__name__ = "gibbs"
        return f
        
    def freeEnergy(self, x):
        pass

    def freeGrad(self, x):
        # => cache face(0) and face(1) linear maps 
        pass


def GBPDiffusion(network):
    pass

def BetheDiffusion(network):
    pass