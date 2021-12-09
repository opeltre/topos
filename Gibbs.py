import topos
import torch

from topos.core import sparse
from topos.core import operators as op
from topos import System 

from torch.distributions import Categorical

from batch import Batch

K = System.closure(['i:j', 'j:k', 'i:k'])
E = K.microstates

def Slice (x, a, b):
    slc = tuple(slice(None) if i in b else x[i] for i in a)
    return slc

def gibbs(H):
    g  = torch.exp(-H + H.min()) 
    return g / g.sum()

def dirac(K, x):
    dx = K.zeros(0)

def dot (w, x, a=None, b=None):
    slc = tuple(slice(None) if i in a else None)

def Condition (K):
    E = K.microstates
    indices = [] 

    for Ea in K[0]:
        a = Ea.key[-1].list()
        for i in range(len(a)):
            Ei = E[a[i]]
            fi = lambda p : Ei.shape.index(Ea.shape.coords(p)[i])
            indices += op.push(Ei, Ea, fi)

    mat = sparse.matrix([E.size, K[0].size], indices)
    conditioning = topos.Linear([K[0], E], mat)
    return conditioning

def Markov (K):
    E    = K.microstates
    cond = Condition(K)
    def markov(H, x):
        """
        Conditional probabilities P(xi|x~i) with P = K.gibbs(H).

        Parameters:
        -----------
            H           :: Field K[0]       Local hamiltonians
            x           :: Field K[0]       Local Dirac masses
        
        Returns:
        --------
            P(xi|x~i)   :: Field K.microstates

        If x is any collection of local observables, 
        the returned field instead computes local
        conditional expectations. 

            markov(H, f)[i](xi) = E[f|xi] 
                                = Sum_{x~i} P(xi|x~i) f(xi, x~i)
        """
        return E.gibbs (cond @ (H * x))
    return markov

def GibbsSampling (K):

    E = K.microstates
    I = list(E.keys())

    C = Condition(K)
    M = Markov(K)

    def sample (H, x0=None, nit=1 << 8, nswap=1):
        
        if x0 == None:
            x0 = E.scalars.field(torch.stack([
                Categorical(torch.ones([Ei.size])).sample() for Ei in E 
            ]).float())

        X     = [x0]
        swaps = torch.randint(E.scalars.size, [nit, nswap])
        Res   = [E.scalars.res([I[j] for j in js]) for js in swaps]

        for js, R in zip(swaps, Res):
            p   = M(H, K.dirac(X[-1]))
            pjs = [p[I[j]] for j in js]
            y   = torch.stack([Categorical(pj).sample() for pj in pjs])
            dy  = R.tgt.field(y.float()) - R @ X[-1]
            X += [X[-1] + R.t() @ dy]

        return (Batch(nit, E.scalars)
                .field(torch.stack([Xi.data for Xi in X[1:]])))

    return sample

G = GibbsSampling(K)
