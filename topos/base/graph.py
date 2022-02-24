from topos.core import sparse, Shape
from topos.io   import alignString, parseTensor

import topos.base.nerve    
import topos.base.complex  

import torch
from torch import stack, cat, arange

class Graph :
    """
    Hypergraphs.

    Given a set of vertices I, a hypergraph is a collection of subsets of I. 
    These subsets are called hyperdeges or more simply cells, regions, faces... 

    The degree or dimension of a face containing n vertices is n + 1.
    A graph is a hypergraph having only faces of dimension <= 1,
    though more precisely it is a simplicial complex of dimension 1.
    """
    
    def __init__(self, *grades, sort=True):
        """ 
        Construct hypergraph from lists of hyperedges by degrees.

        Example:
        -------
            G = Graph([[0], [1], [2]],
                      [[0, 1], [1, 2]])
        """
        
        G = ([parseTensor(js).sort(-1).values for js in grades] 
                if sort else [parseTensor(js) for js in grades])
        if len(G) and G[0].dim() == 1:
            G[0] = G[0].unsqueeze(1)
        
        #--- Adjacency and index tensors ---
        
        A, I, sizes = [], [], [] 
        i, Nvtx  = 0, 1 + max(Gi.max() for Gi in G)

        for k, Gk in enumerate(G):
            
            #--- sort and remove duplicates
            shape = [Nvtx] * (k + 1)
            Ak   = sparse.tensor(shape, Gk).coalesce()
            G[k] = Ak.indices().T
            Nk   = G[k].shape[0]
            A   += [sparse.tensor(shape, Ak.indices(), t=0).coalesce()]
            
            #--- index regions
            idx  = torch.arange(i, i + Nk)
            I   += [sparse.tensor(shape, G[k], idx).coalesce()]
            i   += Nk
            sizes += [Nk]

        #--- Attributes ---
        self.adj   = A
        self.idx   = [sparse.reshape([-1], Ik) for Ik in I]
        self.sizes = sizes
        self.Ntot  = i
        self.Nvtx  = A[0].shape[0]
        self.grades     = G
        self.vertices   = G[0].squeeze(1)
        self.dim        = len(G) - 1
       
    def adjacency(self, k):
        """ Symmetric adjacency tensor in degree k. """
        #--- symmetrize adjacency 
        shape = self.adj[k].size()
        Ak = sparse.tensor(shape, [])
        Sk = SymGroup(k + 1)
        for sigma in Sk:
            Ak += sparse.tensor(shape, self[k].index_select(1, sigma))
        return Ak

    def index (self, js):
        """ Index i of hyperedge [j0, ..., jn]. """
        js = parseTensor(js)
        n  = js.shape[-1]
        I = self.idx[n - 1]
        E = Shape(*([self.Nvtx] * n))
        i = E.index(js)
        return (I.index_select(0, i).to_dense() if i.dim() 
                else I.select(0, i))

    def coords(self, i, d=None):
        """ Hyperedge [j0, ..., jn] at index i. """
        i, begin = parseTensor(i), 0
        if not isinstance(d, type(None)):
            return self[d][i]
        for Gn in self.grades:
            i0 = i[0] if i.dim() == 1 else i
            if i0 - begin < Gn.shape[0]:
                return Gn[i - begin] 
            begin += Gn.shape[0]
    
    def nerve (self, d=-1):
        """ Categorical nerve of the hypergraph. """
        Ntot = self.Ntot 
        N = [torch.ones([Ntot]).to_sparse(),
             self.arrows()]
        arr = [N[1][i].coalesce().indices() for i in range(Ntot)]
        deg = 2
        if d == 1: return N
        while deg != d: 
            ijk = [cat([ij, k]) for ij in N[-1].indices().T \
                                for k in arr[ij[-1]].T]
            if not len(ijk): break
            Nd = sparse.matrix([Ntot] * (deg + 1), stack(ijk))
            N += [Nd.coalesce()]
            deg += 1
        return topos.base.nerve.Nerve(*(Nd.indices().T for Nd in N), sort=False)

    def arrows (self): 
        """ 1-Chains of the hypergraph. """
        Ntot = self.Ntot
        N1   = sparse.matrix([Ntot, Ntot], [])
       
        A    = [sparse.reshape([-1], Ak) for Ak in self.adj]
        E    = [Shape(*Ak.size()) for Ak in self.adj]
        I    = self.idx

        for n, Gn in enumerate(self.grades):
            Nn = Gn.shape[0]

            # row indices
            i_ = I[n].index_select(0, E[n].index(Gn)).to_dense()
            # loop over subfaces
            F  = topos.base.complex.Complex.simplices(Gn)
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

    def __getitem__(self, d):
        """ Degree d hyperedges [j0,...,jd] of the hypergraph. """ 
        return self.grades[d]
    
    def __repr__(self):
        return f"{self.dim} Graph {self}"
    
    def __str__(self):
        return self.show()

    def show (self, json=False):
        s = '{\n\n'
        degree = lambda d: (f'  {d}: ' if not json else f'  "{d}": ')
        s += alignString(degree(0), self[0]) + ',\n\n'
        for d in range(1, self.dim + 1):
            s += alignString(degree(d), self[d]) \
                + (',\n\n' if d < self.dim else '\n\n')
        return s + '}'
