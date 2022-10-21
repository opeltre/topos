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
    pi = pi.view([-1, 2])
    vi = torch.stack([pi[:,1], -pi[:,0]], 1)
    return vi / gmean(pi)[:, None]

def beliefs(pi, pj, eigval):
    pi, pj = pi.view([-1, 2]), pj.view([-1, 2])
    print(pi.shape, eigval.shape)
    sigma = eigval * gmean(pi) * gmean(pj)
    pij = pi[:,:,None] * pj[:,None,:]
    pij += sigma[:,None,None] * torch.tensor([[1, -1], [-1, 1]])
    return pij

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
        return G.Field(1)(p.data[N0:])

    def on_nodes(self, p):
        """ Restrict beliefs to nodes. """
        G = self._classified
        N0 = G.sizes[0]
        return G.Field(0)(p.data[:N0])

    def edge_eigvals(self, p):
        """ Compute belief eigenvalues on edges. """
        G = self._classified
        pij = self.on_edges(p)
        vals = eigval(pij.data)
        return G.scalars().Field(1)(vals)

    def node_eigvecs(self, p):
        """ Return normalized eigenvectors on nodes. """
        G = self._classified
        pi = self.on_nodes(p)
        vecs = eigvec(pi.data)
        return G.Field(0)(vecs.flatten())
    
    def lift_node_beliefs(self, pi, eigvals=1):
        """ 
        Lift node beliefs with prescribed eigenvalues. 
        """
        G = self._classified
        pi = pi.data.view([-1, 2])
        ij = G.grades[1]
        qi = pi[ij[:,0]]
        qj = pi[ij[:,1]]
        if isinstance(eigvals, (int, float)):
            eigvals = eigvals * torch.ones([G.sizes[1]])
        pij = beliefs(qi.data, qj.data, eigvals.data)
        p = torch.cat([pi.data.flatten(), pij.data.flatten()])
        return self.Field(0)(p)

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
        si, sij = s[:n0], s[n0:]
        #--- node beliefs ---
        hi = torch.stack([si, -si], 1)
        pi = torch.exp(-hi)
        pi /= pi.sum([1])[:,None]
        pi = G.Field(0)(pi.flatten())
        #--- full beliefs ---
        p = self.lift_node_beliefs(pi, torch.exp(-sij))
        return p