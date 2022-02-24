from topos.core import sparse, Shape
from topos.io   import alignString, readTensor

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

        G = ([readTensor(js).sort(-1).values for js in grades]
                if sort else [readTensor(js) for js in grades])
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
        self.sizes = torch.tensor(sizes)
        self.begin = torch.tensor([0, *self.sizes.cumsum(0)[:-1]])
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
        js = readTensor(js)
        n  = js.shape[-1]
        I = self.idx[n - 1]
        E = Shape(*([self.Nvtx] * n))
        i = E.index(js)
        return (I.index_select(0, i).to_dense() if i.dim()
                else I.select(0, i))

    def coords(self, i, d=None):
        """ Hyperedge [j0, ..., jn] at index i. """
        i, begin = readTensor(i), 0
        if not isinstance(d, type(None)):
            return self[d][i]
        for Gn in self.grades:
            i0 = i[0] if i.dim() == 1 else i
            if i0 - begin < Gn.shape[0]:
                return Gn[i - begin]
            begin += Gn.shape[0]
    
    def __len__(self):
        """ Maximal degree. """
        return len(self.grades)

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
