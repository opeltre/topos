from topos.core import sparse, Shape, Linear, linear_cache
from topos.io   import readTensor
from .complex   import Complex

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
        return self.zetas(d)[d]

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
            d = max(0, len(self) - d)
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

    @classmethod
    def classify (cls, C, d=-1):
        """ Nerve of a hypergraph (or category). """
        Ntot = C.Ntot
        N0   = torch.ones([Ntot]).to_sparse()
        N1   = sparse.matrix([Ntot, Ntot], C.diagram()._arrows[:,:2].T, t=0)
        N1   = N1.coalesce()
        N = [N0, N1]
        arr = [N1[i].coalesce().indices() for i in range(Ntot)]
        deg = 2
        if d == 1: return N
        while deg != d:
            ijk = [torch.cat([ij, k]) for ij in N[-1].indices().T \
                                for k in arr[ij[-1]].T]
            if not len(ijk): break
            Nd = sparse.matrix([Ntot] * (deg + 1), torch.stack(ijk))
            N += [Nd.coalesce()]
            deg += 1
        return cls((Nd.indices().T for Nd in N), sort=False)

    def __repr__(self):
        return f'{self.dim} Nerve {self}'
