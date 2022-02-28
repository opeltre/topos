from topos.core import sparse, Shape, simplices
from topos.io   import alignString, readTensor
import topos.base.complex

import torch
from torch import stack, cat, arange


class Graph :
    """
    Hypergraphs.

    Given a set of vertices I, a hypergraph is a collection of subsets of I.
    These subsets are called hyperdeges or more simply cells, regions, faces...

    The degree or dimension of a face containing n vertices is n + 1.
    A graph in the usual sense is a hypergraph having only faces of dimension <= 1,
    and more precisely it is a simplicial complex of dimension 1.
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
        self.idx   = I
        self.sizes = torch.tensor(sizes)
        self.begin = torch.tensor([0, *self.sizes.cumsum(0)[:-1]])
        self.Ntot  = i
        self.Nvtx  = A[0].shape[0]
        self.grades     = G
        self.vertices   = G[0].squeeze(1)
        self.dim        = len(G) - 1
        #--- Cache
        self._cache = {}

    def adjacency(self, k):
        """ Symmetric adjacency tensor in degree k. """
        #--- symmetrize adjacency
        shape = self.adj[k].size()
        Ak = sparse.tensor(shape, [])
        Sk = SymGroup(k + 1)
        for sigma in Sk:
            Ak += sparse.tensor(shape, self[k].index_select(1, sigma))
        return Ak

    def index (self, js, output=None):
        """ Index i of hyperedge [j0, ..., jn]. """
        js = readTensor(js)
        n  = js.shape[-1]
        idx = sparse.select(self.idx[n-1], js)
        if output != 'mask':
            return idx
        mask = sparse.mask_index(self.adj[n-1], js)
        return idx, mask

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
   
    def arrows (self):
        """ Strict 1-chains (a > b) of a hypergraph. """
        Ntot = self.Ntot
        arr  = sparse.matrix([Ntot, Ntot], [])
       
        for d, Ad in enumerate(self.grades):
            # number of d-cells
            nd = Ad.shape[0]
            idx_src = sparse.select(self.idx[d], Ad)
            # subfaces
            faces = topos.core.simplices(Ad)
            for k, Bk in enumerate(faces[:-1]):
                # Bk is of shape (nd, nk, k+1):
                nk = Bk.shape[1]
                Bk = Bk.reshape([-1, k+1])
                # Index map
                mask = sparse.index_mask(self.adj[k], Bk)
                src  = idx_src.repeat_interleave(nk)
                tgt  = sparse.select(self.idx[k], Bk)
                AB   = torch.stack([src[mask], tgt[mask]])
                # arrow index pairs
                arr += sparse.matrix([Ntot, Ntot], AB, t=0)
        return arr.coalesce()

    def __len__(self):
        """ Maximal degree. """
        return len(self.grades)
   
    def __iter__(self):
        """ Iterate over grades. """
        return self.grades.__iter__()

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
