from topos.core import sparse

import topos.base.complex
import torch

class Nerve (topos.base.complex.Complex):
    """
    Categorical nerves.

    The nerve of a hypergraph is a simplicial complex.
    The partial order structure of the hypergraph also
    induces a pair of combinatorial automorphisms,
    extending the Zeta transform and Möbius inversion formula
    to the whole complex.

    In degree 0, the zeta transform evaluated at i0 is the sum
    of all evaluations on lower cells j0 <= i0.
    The Möbius transform mu inverts zeta.

    In degree d, the zeta transform evaluated at i0 > ... > id
    sums over all diagrams of the form:

                    i0 > i1 > ... > id
                    >=   >=         >=
                    j0 > j1 > ... > jd

    with the additional constraint that i1 !>= j0, i2 !>= j1, ...
    The Möbius transform extended to degree d inverts zeta.

    References:
    -----------
    - Rota, 1964:
        On the Foundations of Combinatorial Theory - I. Theory of Möbius Transforms
    - Peltre, 2020:
        Message-Passing Algorithms and Homology, Chapter III,
        https://arxiv.org/abs/2009.11631
    """
   
    def zeta (self, d):
        """ Degree-d zeta transform.

            See Nerve.zetas(d) to keep graded components until d. 
        """
        return self.zetas(d)[d]

    def zetas (self, d):
        """ List of zeta transforms in degrees <= d. """

        zt, N = [], [Nd.shape[0] for Nd in self.grades]

        # strict inclusions
        A  = self.adj[1]
        # weak inclusions
        zt0 = ( sparse.matrix([N[0], N[0]], A.indices(), t=False)
              + sparse.eye(N[0])).coalesce()
       
        zt += [zt0]
       
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

    def __repr__(self):
        return f'{self.dim} Nerve {self}'
