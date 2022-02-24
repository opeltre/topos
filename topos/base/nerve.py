from topos.core import sparse, Shape
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
            zeta(y)[a0 > ... > ad] = sum [y[b0 > ... > bd] | 
                                           b0 <= a0,  ..., bd <= ad,
                                           b0 !<= a1, ..., bd_1 !<= ad=
        """
        if d < 0 : d = max(0, len(self) - d)
        N = [Nd.shape[0] for Nd in self.grades]
        # strict inclusions : adj[1]
        A  = self.adj[1]
        # weak inclusions   : zeta[0]
        zt0 = ( sparse.matrix([N[0], N[0]], A.indices(), t=False)
              + sparse.eye(N[0])).coalesce()
        # full automorphism
        zt = [zt0]

        def next_diagrams (chain_i, chain_j):
            """ Extend diagrams to the right:

                    i0 > i1 > ... > id
                    >=   >=         >=
                    j0 > j1 > ... > jd
            """
            order = sparse.reshape([-1], zt0).coalesce()
            # Extend chain_i
            below_i = sparse.index_select(A, chain_i[-1])
            next_i  = below_i.indices()[1]
            deg_i   = sparse.sum_dense(below_i, [1]).long()
            chain_j = chain_j.repeat_interleave(deg_i, 1)
            chain_i = chain_i.repeat_interleave(deg_i, 1)
            chain_i = torch.cat([chain_i, next_i[None,:]])
            # Extend chain_j
            below_j = sparse.index_select(A, chain_j[-1])
            next_j  = below_j.indices()[1]
            deg_j   = sparse.sum_dense(below_j, [1]).long()
            chain_i = chain_i.repeat_interleave(deg_j, 1)
            chain_j = chain_j.repeat_interleave(deg_j, 1)
            chain_j = torch.cat([chain_j, next_j[None,:]])
            # Check that i[d] >= j[d] and i[d] !>= j[d-1]
            idx1  = N[0] * chain_i[-1] + chain_j[-1]
            idx2  = N[0] * chain_i[-1] + chain_j[-2]
            mask1 =   sparse.index_mask(order, idx1)
            mask2 = ~ sparse.index_mask(order, idx2)
            nz = (mask1 * mask2).nonzero().flatten()
            # Return diagrams
            return chain_i[:, nz], chain_j[:, nz]

        acc = [zt0.indices().unsqueeze(1)]

        for k in range(d):
            acc += [torch.stack(next_diagrams(*acc[-1]))]

        for chains, n, begin in zip(acc[1:], N[1:d+1], self.sizes[:d]):
            ij = torch.stack([self.index(chains[0].T) - begin,
                              self.index(chains[1].T) - begin])
            zt += [sparse.tensor([n, n], ij, t=False)]

        return zt

    @classmethod
    def nerve (cls, G, d=-1):
        """ Categorical nerve of a hypergraph. """
        Ntot = G.Ntot
        N = [torch.ones([Ntot]).to_sparse(),
             cls.arrows(G)]
        arr = [N[1][i].coalesce().indices() for i in range(Ntot)]
        deg = 2
        if d == 1: return N
        while deg != d:
            ijk = [torch.cat([ij, k]) for ij in N[-1].indices().T \
                                for k in arr[ij[-1]].T]
            if not len(ijk): break
            Nd = sparse.matrix([Ntot] * (deg + 1), torch.stack(ijk))
            N += [Nd.coalesce()]
            deg += 1
        return cls(*(Nd.indices().T for Nd in N), sort=False)

    @staticmethod
    def arrows (G):
        """ 1-Chains of a hypergraph. """
        Ntot = G.Ntot
        N1   = sparse.matrix([Ntot, Ntot], [])

        A    = [sparse.reshape([-1], Ak) for Ak in G.adj]
        E    = [Shape(*Ak.size()) for Ak in G.adj]
        I    = G.idx

        for n, Gn in enumerate(G.grades):
            Nn = Gn.shape[0]
            # row indices
            i_ = I[n].index_select(0, E[n].index(Gn)).to_dense()
            # loop over subfaces
            F  = Complex.simplices(Gn)
            for k, Fk in enumerate(F[:-1]):
                # valid column indices
                nz = (A[k].index_select(0, E[k].index(Fk))
                          .to_dense()
                          .view([-1, Nn])
                          .nonzero())
                j_ = (I[k].index_select(0, E[k].index(Fk))
                          .to_dense()
                          .view([-1, Nn]))
                # ordered pairs
                ij = torch.tensor([[i_[y], j_[x, y]] for x, y in nz])
                N1 += sparse.matrix([Ntot, Ntot], ij)

        chains = N1.coalesce().indices()
        return sparse.matrix([Ntot, Ntot], chains, t=0).coalesce()

    def __repr__(self):
        return f'{self.dim} Nerve {self}'
