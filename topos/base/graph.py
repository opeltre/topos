from .quiver    import Quiver
from topos.core import sparse, Shape, simplices

import topos.io as io
import topos.base.nerve 
import torch

from .functor import Functor
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
        super().__init__(grades, functor, sort)     

    def adjacency(self, k):
        """ Symmetric adjacency tensor in degree k. """
        #--- symmetrize adjacency
        shape = self.adj[k].size()
        Ak = sparse.tensor(shape, [])
        Sk = SymGroup(k + 1)
        for sigma in Sk:
            Ak += sparse.tensor(shape, self[k].index_select(1, sigma))
        return Ak

    def fmap (self, f):
        """
        Functorial map associated to f = [a, b] or f = [[*as], [*bs]]
        """
        
    
    def pull(self, functor): 
        """ 
        Pre-compose a functor by the index to coordinates mapping. 
        """
        def obj(i): 
            return functor(self.coords(i))

        def fmap(f):
            src, tgt = ob.gradej(f[0]), obj(f[1])
            a, b = self.coords(f[0]).contiguous(), self.coords(f[1]).contiguous()
            return tgt.index @ functor.fmap((a, b)) @ src.coords
    
    def push(self, functor):
        """
        Post-compose by a functor on local objects.
        """
        pass

    def quiver(self):
        """
        Quiver of strict 1-chains a > b for inclusion.

        The restriction maps G.obj(a) -> G.obj(b) are encoded 
        as edge labels i.e. edges q of Q[1] are of the form: 

            q = [a, b, 1 + j0, ..., 1 + jk, 0, ..., 0] 

        where j0, ..., jk are the position (< a.dim) of 
        forgotten indices in the restriction a -> b.
        """
        # Cache quiver
        if not isinstance(self._quiver, type(None)):
            return self._quiver

        # Functor valued quiver
        if not self.trivial:
            T = self.scalars()
            Q = T.quiver()
            def obj (a): return a[0] if a.dim() else a
            def hom (f): return f    
            lift = Functor(lambda a: a[0] if a.dim() else a,
                           lambda f: f)
            F = self.functor @ T.Coords @ lift
            return Quiver(Q, F)

        # Base quiver i.e. scalar valued
        Ntot = self.Ntot
        Q1  = sparse.matrix([Ntot, Ntot], [])
        fibers = []
        maps = []

        for d, Gd in enumerate(self.fibers):
            # Source d-cells
            Ad = Gd.keys
            nd = Ad.shape[0]
            idx_src = sparse.select(self.idx[d], Ad) + self.begin[d]
            # subfaces and forgotten indices
            faces = simplices(Ad)
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
                Q1 += sparse.matrix([Ntot, Ntot], AB.T)

        edges = Q1.coalesce().indices().T
        Q = Quiver([torch.arange(Ntot), edges])
        self._quiver = Q   
        return self._quiver

    def __repr__(self):
        return f"{self.__name__}"

    def nerve(self):
        return topos.base.nerve.Nerve.classify(self)