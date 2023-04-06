from topos.base import FreeFunctor
from .network import Network
import fp
import torch

binaryFunctor = FreeFunctor(2)

Ising = FreeFunctor(2)

def eigval(pij):
    pij = pij.view([-1, 2, 2])
    pi, pj = pij.sum([2]), pij.sum([1])
    num = ((pij[:,0, 0] * pij[:, 1, 1]) 
          - (pij[:,0, 1]) * pij[:, 1, 0])
    return num / (gmean(pi) * gmean(pj))

def gmean(pi):
    return (pi[:,0] * pi[:,1]).sqrt()

def eigvec(pi):
    Ns = pi.shape[:-1]
    pi = pi.view([-1, 2])
    vi = torch.stack([pi[:,1], -pi[:,0]], 1)
    vi = vi / gmean(pi)[:, None]
    return vi.view([*Ns, -1, 2])

def beliefs(pi, pj, eigval):
    Ns = pi.shape[:-1]
    pi, pj = pi.view([-1, 2]), pj.view([-1, 2])
    sigma = eigval.view([-1]) * gmean(pi) * gmean(pj)
    pij = pi[:,:,None] * pj[:,None,:]
    pij += sigma[:,None,None] * torch.tensor([[1, -1], [-1, 1]])
    return pij.view([*Ns, 2, 2])

for method in [eigval, eigvec, gmean, beliefs]:
    setattr(Ising, method.__name__, method)


class IsingNetwork(Network):
    """ 
    Network subclass for binary graphs. 
    
    N.B. This class assumes that the `Nerve` instance 
    classifies a 1-dimensional `Complex` (i.e. a graph) 
    with binary variables on each node. 
    """

    def on_edges(self, p):
        """ Restrict beliefs to edges. """
        G = self._classified
        N0 = G.sizes[0]
        slc = [slice(None)] * (p.dim() - 1)
        pij = p.data[(*slc, slice(N0, None))]
        return G[1].field(pij.contiguous())

    def on_nodes(self, p):
        """ Restrict beliefs to nodes. """
        G = self._classified
        N0 = G.sizes[0]
        slc = [slice(None)] * (p.dim() - 1)
        pi  = p.data[(*slc, slice(0, N0))]
        return G[0].field(pi.contiguous())

    def edge_eigvals(self, p):
        """ Compute belief eigenvalues on edges. """
        G = self._classified
        pij = self.on_edges(p)
        Ns = pij.shape[:-1]
        vals = eigval(pij.data.view([-1, G.sizes[1]]))
        return G[1].scalars().field(vals.view([*Ns, -1]))

    def node_eigvecs(self, p):
        """ Return normalized eigenvectors on nodes. """
        G = self._classified
        pi = self.on_nodes(p)
        vecs = eigvec(pi.data).flatten(start_dim=-2)
        return G[0].field(vecs)
    
    def lift_node_beliefs(self, pi, eigvals=1):
        """ 
        Lift node beliefs with prescribed eigenvalues. 
        """
        #--- batch shape
        Ns = pi.shape[:-1]
        slc = ([slice(None)] * len(Ns))
        #--- split nodes
        G = self._classified
        pi = pi.data.view([*Ns, -1, 2])
        ij = G.grades[1]
        #--- source and target beliefs 
        qi = pi[(*slc, ij[:,0])]
        qj = pi[(*slc, ij[:,1])]
        #--- lift couplings to pairwise beliefs
        if isinstance(eigvals, (int, float)):
            eigvals = eigvals * torch.ones([G.sizes[1]])
        pij = beliefs(qi.data, qj.data, eigvals.data)
        #--- return batched beliefs
        p = torch.cat([
            pi.data.flatten(start_dim=-2), 
            pij.data.flatten(start_dim=-3)
        ], -1)
        return self[0].field(p)

    def lift_interaction(self, s):
        """ 
        Lift scalar field to consistent beliefs. 
        
        The components (si, sij) of s represent local magnetic
        fields and local magnetic couplings respectively. 
        Node beliefs are lifted as `pi(xi) = exp(-si.xi)`,
        edge beliefs will have `exp(-sij)` as eigenvalues. 
        """
        G = self._classified
        n0, n1 = G.scalars().sizes[:2]
        s = s.data
        last = [slice(None)] * (s.dim() - 1)
        si  = s[(*last, slice(0, n0))]
        sij = s[(*last, slice(n0, None))]
        #--- check shapes
        if sij.shape[-1] != n1 or si.shape[-1] != n0:
            raise RuntimeError(
            f'wrong interaction shape: {s.shape} for sizes {(n0, n1)}')
        #--- node beliefs ---
        hi = torch.stack([si, -si], -1)
        pi = torch.exp(-hi)
        pi /= pi.sum([-1])[(*last, slice(None), None)]
        pi = G[0].field(pi.flatten(start_dim=-2))
        #--- full beliefs ---
        p = self.lift_node_beliefs(pi, torch.exp(-sij))
        return p