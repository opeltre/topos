from .quiver    import Quiver
from topos.core import sparse, Shape, simplices
from topos.io   import alignString, readTensor

import topos.base.nerve 
import torch

from .multigraph import MultiGraph


class Graph (MultiGraph):
    """
    Hypergraphs.

    Given a set of vertices I, a hypergraph is a collection of subsets of I.
    These subsets are called hyperdeges or more simply cells, regions, faces...

    The degree or dimension of a face containing n vertices is n + 1.
    A graph in the usual sense is a hypergraph having only faces of dimension <= 1,
    and more precisely it is a simplicial complex of dimension 1.
    """

    def __init__(self, grades, functor=None, sort=True):
        """
        Construct hypergraph from lists of hyperedges by degrees.

        Hyperedges in each degree are given to `Sheaf.sparse` in order
        to create graded index sheaves with sparse adjacency tensors.

        Example:
        -------
            G = Graph([[0], [1], [2]],
                      [[0, 1], [1, 2]])
        """
        G = ([readTensor(js).sort(-1).values for js in grades]
                if sort else [readTensor(js) for js in grades])
        super().__init__(G, functor)
        self._quiver = None

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
        #--- Degree access
        if isinstance(js, int):
            return js
        #--- Index of hyperedge batch
        js = readTensor(js)
        n  = js.shape[-1] - 1
        idx = sparse.select(self.idx[n], js)
        #--- Graded fiber offset
        offsets = self.sizes.cumsum(0).roll(1)
        offset  = offsets[n] if n > 0 else 0
        #--- Return index only
        if output == None:
            return idx + offset
        #--- Keep mask
        mask = sparse.index_mask(self.adj[n], js)
        return idx + offset, mask

    def coords(self, i, d=None):
        """ Hyperedge [j0, ..., jn] at index i. """
        i, begin = readTensor(i), 0
        if not isinstance(d, type(None)):
            return self.grades[d][i]
        for Gn in self.fibers:
            i0 = i[0] if i.dim() == 1 else i
            if i0 - begin < Gn.keys.shape[0]:
                return Gn.keys[i - begin]
            begin += Gn.keys.shape[0]

    def arrow (self, a, b):
        """
        Return a size G[da] x G[db] matrix representing functorial maps.

        Inputs:
            - a: tensor of shape (da) or (N, da)
            - b: tensor of shape (db) or (N, db)
        """
        da, db = a.shape[-1] - 1, b.shape[-1] - 1
        shape = self[da].size, self[db].size
        ij = torch.stack([self[da].index(a), self[db].index(b)])
        if self.trivial:
            return sparse.matrix(shape, ij, t=False)
    
    def quiver(self):
        """
        Quiver of strict 1-chains a > b for inclusion.

        The restriction maps G.obj(a) -> G.obj(b) are encoded 
        as edge labels i.e. edges q of Q[1] are of the form: 

            q = [a, b, 1 + j0, ..., 1 + jk, 0, ..., 0] 

        where j0, ..., jk are the position (< a.dim) of 
        forgotten indices in the restriction a -> b.
        """
        if not isinstance(self._quiver, type(None)):
            return self._quiver
        
        Ntot = self.Ntot
        Nlbl = [2 + self.dim] * self.dim
        Q1  = sparse.matrix([Ntot, Ntot, *Nlbl], [])
        fibers = []
        maps = []

        for d, Gd in enumerate(self.fibers):
            # Source d-cells
            Ad = Gd.keys
            nd = Ad.shape[0]
            idx_src = sparse.select(self.idx[d], Ad) + self.begin[d]
            # subfaces and forgotten indices
            faces, indices = simplices(Ad, True)
            Js = []
            # Target k-cells
            for k, Bk in enumerate(faces[:-1]):
                # Bk is of shape (nd, nk, k+1):
                nk = Bk.shape[1]
                Bk = Bk.reshape([-1, k+1])
                # Index map
                mask = sparse.index_mask(self[k].adj, Bk)
                src  = idx_src.repeat_interleave(nk)
                tgt  = sparse.select(self[k].idx, Bk) + self.begin[k]
                AB   = torch.stack([src[mask], tgt[mask]])
                # Label edges by forgotten indices
                js  = torch.tensor(indices[k])
                pad = torch.zeros([self.dim - js.shape[-1]]).repeat(js.shape[0], 1)
                label = torch.cat([1 + js, pad], dim=-1).repeat(nd, 1)
                edges = torch.cat([AB.T, label[mask]], dim=-1)
                # arrow index pairs
                shape = [Ntot, Ntot, *Nlbl]
                Q1 += sparse.matrix(shape, edges)

        edges = Q1.coalesce().indices().T
        Q = Quiver(torch.arange(Ntot), edges)
        self._quiver = Q if self.trivial else Quiver(Q.grades, self.functor)
        return self._quiver

    def __repr__(self):
        return f"{self.dim} Graph {self}"

    def nerve(self):
        return topos.base.nerve.Nerve.classify(self.quiver())