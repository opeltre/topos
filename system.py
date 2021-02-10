import torch 

from set import Set
from dict import Dict

from hypergraph import Hypergraph

from tensor import Tensor, Product, Matrix
from functional import Id, Functional, Operator


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
        zeta = I + self.below 

        ## Higher degree zeta & mu transforms

        ## zeta_n = zeta_n-1 >> zeta
        self.I = self.rpows(I)
        self.zeta = self.rpows(zeta)

        ## mu_n = zeta_n ** (-1)
        self.mu = self.zeta.map(
            lambda zn, n: self.invert(zn, n)
        )
        
        # Differential and codifferential

        self.d = Product(self.diff(i) for Ni, i in self.N)
        self.delta = Product(self.codiff(i) for Ni, i in self.N)
        d = sum(di for di, i in self.d)
        L = d @ d.t() + d.t() @ d
        self.Laplacian = L

        # Functors 

        I0 = self.I[0].uncurry() 

        ## cylindrical extensions
        self.J = I0 + Functional({
            (a, b): self.extend(a, b) for a, b in chains
        })
        ## marginal projections
        self.Sigma = I0 + Functional({
            (a, b): self.project(a, b) for a, b in chains
        })
        ## effective energies 
        self.F = I0 + Functional({
            (a, b): self.effective(a, b) for a, b in chains
        })
        ## free energies
        self.F += Functional({
            (a,):   self.effective(a) for a in self
        })

        ## Operators 
        OpJ = lambda t: Operator(t, fmap=self.J)
        OpF = lambda t: Operator(t, cofmap=self.F)
        OpS = lambda t: Operator(t, cofmap=self.Sigma)

        self.Zeta   = self.zeta.fmap(OpJ)
        self.Mu     = self.mu.fmap(OpJ)
        self.Delta  = self.delta.fmap(OpJ)
        self.D      = self.d.fmap(OpS)
        self.Deff   = self.d.fmap(OpF)


    def field (self, *args):
        return Tensor(*args)
    
    def zeros(self, n): 
        zero = lambda _, a: torch.zeros(self.shape[a[0]])
        return self.N[n].map(zero)
        
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

    def effective(self, a, b=None): 
        if not b or len(b) == 0:
            F_a = lambda ha: - torch.logsumexp(-ha)
            F_a.__name__ = f"Free Energy {a}"
            return F_a
        S_ab = self.project(a, b)
        F_ab = lambda ha: - torch.log(S_ab(torch.exp(-ha)))
        F_ab.__name__ = f"Effective Energy {a} > {b}"
        return F_ab

    def gaussian(self, n):
        gauss = lambda _, a: torch.randn(self.shape[a[0]])
        return self.N[n].map(gauss)

    def diff(self, n): 
        if not n < len(self.N) - 1:
            return Matrix()
        face = lambda i: Matrix({
            a : {a.forget(i) : 1} for _, a in self.N[n + 1]
        })
        d = sum((-1)**i * face(i) for i in range(n + 2))
        return d

    def codiff(self, n): 
        if not n > 0:
            return Matrix()
        return self.diff(n - 1).t()

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
