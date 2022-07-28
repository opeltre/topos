from topos.core import sparse, Shape, Linear, linear_cache
import topos.io as io

from .functor   import Functor
from .complex   import Complex
from .graph     import Graph

import torch

class Nerve (Complex):
    """
    Categorical nerves.

    The nerve of a hypergraph is a simplicial complex.
    The partial order structure of the hypergraph also
    induces a pair of combinatorial automorphisms,
    extending the Zeta transform and Möbius inversion formula
    to the whole complex.

    In degree 0, the zeta transform evaluated at a0 is the sum
    of all evaluations on lower cells b0 <= a0.
    The Möbius transform mu inverts zeta.

    In degree d, the zeta transform evaluated at a0 > ... > ad
    sums over all diagrams of the form:

                    a0 > a1 > ... > ad
                    >=   >=         >=
                    b0 > b1 > ... > bd

    with the additional constraint that a1 !>= b0, a2 !>= b1, ...
    The Möbius transform extended to degree d inverts zeta.

    References:
    -----------
    - Rota, 1964:
        On the Foundations of Combinatorial Theory - I. Theory of Möbius Transforms
    - Peltre, 2020:
        Message-Passing Algorithms and Homology, Chapter III,
        https://arxiv.org/abs/2009.11631
    """
    
    @linear_cache("zeta", "\u03b6")
    def zeta (self, d=0):
        """ 
        Degree-d zeta transform.
        
        Recursive function. 
        Calls `Nerve.zetas(d)` to compute graded components up to degree d.
        """
        if self.trivial:
            return self.zetas(d)[d]
        # extend scalar valued transform
        O = self.scalars()
        zt = O.zeta(d)
        i, j = O.begin[d] + zt.data.indices()
        a, b = self.coords(i), self.coords(j)
        Fab  = self.fmap([a, b])
        N = self[d].size
        zt = sparse.matrix([N, N], Fab - self.begin[d], t=False)
        return Linear(self[d], self[d])(zt, 0, "\u03b6")

    def zetas (self, d=-1):
        """
        List of zeta transforms in degrees <= d.
    
            #--- degree 0
            zeta(x)[a]       = sum [x[b] | b <= a]
            #--- degree 1
            zeta(m)[a0 > a1] = sum [m[b0 > b1] | b0 <= a0, b1 <= a1, b0 !<= a1]
            #--- ...
            #--- degree d
            zeta(y)[a0 > ... > ad] =
                sum [y[b0 > ... > bd] | b0 <= a0,  ..., bd <= ad,
                                        b0 !<= a1, ..., bd_1 !<= ad]
        """
        if d < 0 : 
            d = max(0, len(self.grades) - 1)
        if not self.trivial:
            return [self.zeta(k) for k in range(d)]
        # sizes
        N = [Nd.keys.shape[0] for Nd in self.fibers]
        # strict inclusions : adj[1]
        A  = self.adj[1]
        # weak inclusions   : zeta[0]
        zt0 = ( sparse.matrix([N[0], N[0]], A.indices(), t=False)
              + sparse.eye(N[0])).coalesce()
        # full automorphism
        zt = [zt0]

        def next_diagrams (chain_a, chain_b):
            """ Extend diagrams to the right:

                    a0 > a1 > ... > ad
                    >=   >=         >=
                    b0 > b1 > ... > bd
            """
            order = sparse.reshape([-1], zt0).coalesce()
            # Extend chain_a
            below_a = sparse.index_select(A, chain_a[-1])
            next_a  = below_a.indices()[1]
            deg_a   = sparse.sum_dense(below_a, [1]).long()
            chain_b = chain_b.repeat_interleave(deg_a, 1)
            chain_a = chain_a.repeat_interleave(deg_a, 1)
            chain_a = torch.cat([chain_a, next_a[None,:]])
            # Extend chain_b
            below_b = sparse.index_select(A, chain_b[-1])
            next_b  = below_b.indices()[1]
            deg_b   = sparse.sum_dense(below_b, [1]).long()
            chain_a = chain_a.repeat_interleave(deg_b, 1)
            chain_b = chain_b.repeat_interleave(deg_b, 1)
            chain_b = torch.cat([chain_b, next_b[None,:]])
            # Check that i[d] >= j[d] and i[d] !>= j[d-1]
            idx1  = N[0] * chain_a[-1] + chain_b[-1]
            idx2  = N[0] * chain_a[-1] + chain_b[-2]
            mask1 =   sparse.index_mask(order, idx1)
            mask2 = ~ sparse.index_mask(order, idx2)
            nz = (mask1 * mask2).nonzero().flatten()
            # Return diagrams
            return chain_a[:, nz], chain_b[:, nz]

        acc = [zt0.indices().unsqueeze(1)]

        for k in range(d):
            acc += [torch.stack(next_diagrams(*acc[-1]))]

        for Gk, chains, n in zip(self.fibers[1:], acc[1:], N[1:d+1]):
            zt += [Gk.arrow(chains[0].T, chains[1].T)]

        out = []
        for d, ztd in enumerate(zt):
            lin = Linear(self[d], self[d])(ztd, degree=0, name="\u03b6")
            out += [lin]
            self[d]._cache["zeta"] = lin
        return out

    def fmap(self, f):
        """
        Functor graph over chains [a0, ..., ak], [b0, ..., br] with ak -> br.
        """
        Q = self.quiver()
        a, b = io.readTensor(f[0]), io.readTensor(f[1])
        if a.dim() == 1: 
            a, b = a.unsqueeze(0), b.unsqueeze(0)
        k, r = a.shape[-1] - 1, b.shape[-1] - 1
        i, j = self[k].index(a), self[r].index(b)
        ak, br = (a[-1], b[-1]) if a.dim() == 1 else (a[:,-1], b[:,-1])
        # outside identities
        eq = (ak == br)
        if (~eq).sum() > 0:
            Qab = Q.fmap(torch.stack([ak[~eq], br[~eq]]))
            Fa, Fb = torch.zeros([2, ak[~eq].shape[0]])
            off_a = Q[0].begin[ak[~eq]] - self.begin[k] - self[k].begin[i[~eq]]
            off_b = Q[0].begin[br[~eq]] - self.begin[r] - self[r].begin[j[~eq]] 
            Fa = Qab[0] - off_a.repeat_interleave(Q[0].sizes[ak[~eq]])
            Fb = Qab[1] - off_b.repeat_interleave(Q[0].sizes[ak[~eq]])
            Fab = torch.stack([Fa, Fb])
        else: 
            Fab = torch.zeros([2, 0])
        # identities 
        if eq.sum() > 0:
            ns = Q[0].sizes[ak[eq]]
            off_eq = Q[0].sizes[ak[eq]].cumsum(0).roll(1)
            off_eq[0] = 0
            off_a = off_eq.repeat_interleave(ns)
            begin_a = self.begin[k] + self[k].begin[i[eq]].repeat_interleave(ns)
            begin_b = self.begin[r] + self[r].begin[j[eq]].repeat_interleave(ns)
            n = off_a.shape[0]
            Fa = torch.arange(n) - off_a + begin_a
            Fb = torch.arange(n) - off_a + begin_b
            Faa = torch.stack([Fa, Fb])
        else:
            Faa = torch.zeros([2, 0])
        # total graph
        return torch.cat([Fab, Faa], dim=1)

    @classmethod
    def classify (cls, quiver, d=-1):
        """ Nerve of a quiver (e.g. hypergraph ordering or category). """
        if isinstance(quiver, Graph):
            name = quiver.__name__
            quiver = quiver.quiver()
            quiver.__name__ = name
        # vertices and arrows
        Ntot = quiver[0].keys.shape[0]
        Q1 = quiver.grades[1]
        N0   = torch.ones([Ntot]).to_sparse()
        N1   = sparse.matrix([Ntot, Ntot], Q1[:,:2])
        N1   = N1.coalesce()
        # nerve Nd, d <= 1
        N = [N0, N1]
        arr = [N1[i].coalesce().indices() for i in range(Ntot)]
        deg = 2
        if d == 1: return N
        # nerve Nd, d > 1
        while deg != d:
            ijk = [torch.cat([ij, k]) for ij in N[-1].indices().T \
                                for k in arr[ij[-1]].T]
            if not len(ijk): break
            Nd = sparse.matrix([Ntot] * (deg + 1), torch.stack(ijk))
            N += [Nd.coalesce()]
            deg += 1
       
        # scalar coefficients
        if quiver.trivial:
            F = None
        # functorial coefficients
        else:
            def last_obj(a):
                a = io.readTensor(a, dtype=torch.long)
                return a[-1] if a.dim() == 1 else a[:,-1]
            def last_hom(f):
                a, b = f[0], f[1]
                return last_obj(a), last_obj(b)
            last = Functor(last_obj, last_hom)
            F = quiver.functor @ last

        NQ = cls((Nd.indices().T for Nd in N), functor=F, sort=False)
        NQ.__name__ = (f'N({quiver.__name__})' 
                       if '__name__' in dir(quiver) else 'N(Q)')
        NQ._quiver = quiver
        return NQ

    def __repr__(self):
        name = self.__name__ if '__name__' in dir(self) else 'N'
        return f'{name}'