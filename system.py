import torch 

from set import Set
from dict import Dict

from hypergraph import Hypergraph

from tensor import Tensor, Product
from functional import Functional, Id


class System (Hypergraph):

    def __init__(self, K, shape=2):

        if not isinstance(K, Hypergraph):
            K = Hypergraph(K)
        if not isinstance(K, System):
            K = K.closure()
        super().__init__((a for a in K))

        ## E(i) = number of microstates for atom i
        E = lambda i: shape if type(shape) == int else shape[i]
        self.shape = Dict({a: Product(E(i) for i in a) for a in self})
        
        ## Nerve(i) = { a0 > ... > ai | a in K**i }  
        chains = [(a, b) for a, b in K * K if a > b]
        N0 = self.field({
            (a, ) : 1 for a in K
        })
        N1 = self.field({
            (a, b) : 1 for a, b in chains
        })
        self.N = N0.cup_pows(N1)

        # Combinatorics

        I = self.field({
            (a, a) : 1 for a in K
        }).curry()
        self.below = N1.curry(0)
        self.above = N1.curry(1)

        ## zeta_ab = 1 if a >= b else 0
        self.zeta = I + self.below 

        ## Zeta_n = Zeta_n-1 >> zeta
        self.I = self.rpows(I)
        self.Zeta = self.rpows(self.zeta)

        ## Mu_n = Zeta_n ** (-1)
        self.Mu = self.Zeta.map(
            lambda zn, n: self.invert(zn, n)
        )

        # Functors 
        self.J = self.I[0].uncurry() + Functional({
            (a, b): self.extend(a, b) for a, b in chains
        })
        self.Sigma = self.I[0].uncurry() + Functional({
            (a, b): self.project(a, b) for a, b in chains
        })

    def extend(self, a, b):
        pull = [slice(None) if i in b else None for i in a]
        J_ab = lambda tb: tb[pull]
        J_ab.__name__ = f"Extend {a} < {b}"
        return J_ab

    def project(self, a, b):
        dim = tuple((i for i, x in enumerate(a) if x not in b))
        S_ab = lambda qa: torch.sum(qa, dim=dim)
        S_ab.__name__ = f"Sum {a} > {b}"
        return S_ab

    def field (self, *args):
        return Tensor(*args)
    
    def zeros(self, n): 
        return self.N[n].map(
            lambda _, a: torch.zeros(self.shape[a[0]])
        )
        
    def gaussian(self, n):
        return self.N[n].map(
            lambda _, a: torch.randn(self.shape[a[0]])
        )

    def rtimes(self, t, s):
        r = t.__class__()
        s = s.fibers(lambda sb, b: b.project(0))
        cone = self.below
        times = lambda ti, sj: self.rtimes(ti, sj) if\
            isinstance(ti, Tensor) and isinstance(sj, Tensor) \
            else ti * sj
        for ta, a in t:
            i = a.p(-1)
            for _, j in cone[i] if i in cone else []:
                for sb, b in s[j] if j in s else []:
                    r[a|b] = times(ta, sb)
        return r.trim()
        
    def rpows(self, t, s=None, N=10):
        s = s if s else t
        pows, pk, i = [], t, 0
        while not Tensor.iszero(pk) and i <= N:
            pows += [pk]
            pk = self.rtimes(pk, s)
            i += 1
        return Product(pows)

    def invert(self, matrix, n=0, N=10):
        h = matrix - self.I[n]
        pows, pk, i = [], self.I[n], 0
        while not Tensor.iszero(pk) and i <= N:
            pows += [pk]
            pk = pk @ h
            i += 1
        return sum((-1)**k * pk for pk, k in Product(pows))
    
    def __getitem__(self, n): 
        if type(n) == tuple:
            Kn = (self.nerve(ni) for ni in n)
            return [a for a in product(*Kn)]
        return self.nerve(n)
    
    def __repr__(self): 
        elems = [str(e) for e in self]
        s = ' '.join(elems)
        return f"System {s}"
