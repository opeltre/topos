from .sheaf     import Sheaf
from .functor   import Functor

from topos.core import sparse, Shape, simplices
import topos.io as io
from topos.io   import alignString, readTensor

import torch
import fp

class MultiGraph (Sheaf):
    """
    (Directed hyper)-multigraphs

    This base class uses graded integer tensors as keys, allowing to use
    the sparse backend for efficient topological computations.
    
    N.B: No topology is explicitly assumed in this class, 
    and only the object map of the `functor` argument is used.
    For cases where `functor.fmap` is applied on a certain choice
    of arrows, see subclasses e.g. Quiver, Graph, Complex, Nerve...
    """

    def __init__(self, grades, functor=None, sort=False):
        """
        Create multigraph G from list of graded multiedges.

        A degree-d multiedge is understood as an integer tensor

            gd = [i0, ..., id, b1, ..., bk]

        where `i0, ..., id` represent vertices in {0, ..., Nvtx} 
        (note that `G[0]` does not have to contain all vertices)
        and `b1, ..., bk` represent edge labels.

        If `functor` is a list, then each element will be passed 
        to the constructor of the corresponding graded fiber. 
        Otherwise, the same `functor` argument is passed to 
        all graded constructors. 

        Example:
        -------
        >>> G = MultiGraph([
                [[0], [1], [2]], 
                [[0, 1], [1, 2]],
                [[1, 2, 3]]
            ])

        >>> FG = MultiGraph(G, functor=lambda g : [2] * len(g))
        """
        if isinstance(grades, MultiGraph):
            G = grades.grades
        else:
            G = [readTensor(js) for js in grades] 
        if len(G) and G[0].dim() == 1:
            G[0] = G[0].unsqueeze(1)

        self.grades  = G
        self.dim     = len(G) - 1

        #--- Label sizes
        nlabels = [Gi.shape[-1] - (i + 1) for i, Gi in enumerate(G)]
        self.nlabels = nlabels

        #--- Sparse fibers ---
        Nvtx  = int(1 + max(Gi[:,:i+1].max() for i, Gi in enumerate(G)))
        Nlbl  = [(int(1 + Gi[:,i+1:].max()) if nlabels[i] else 1) for i, Gi in enumerate(G)]
        shapes = [[Nvtx] * (d + 1) + [Nlbl[d]] * nlabels[d] for d in range(len(G))]
        fibers = [Sheaf.sparse(shapes[d], G[d], functor, degree=d) for d in range(len(G))]
        super().__init__(functor=fibers)

        #--- Graph attributes ---
        self.adj   = [Gd.adj for Gd in fibers]
        self.idx   = [Gd.idx for d, Gd in enumerate(fibers)]
        self.Ntot  = sum(Gd.keys.shape[0] for Gd in fibers)
        self.size  = self.sizes.sum()
        self.Nvtx  = self.adj[0].shape[0]
        self.vertices   = G[0].squeeze(1)

        #--- Index <-> Coordinate functors ---
        self.functor = functor
        self.Coords  = Functor(self.coords, self.coords_fmap)
        self.Index   = Functor(self.index, self.index_fmap)
        
        self._quiver = None
        self.__name__ = 'G'

    def scalars(self):
        return self if self.trivial else self.__class__(self.grades)

    def __getitem__(self, d):
        """ Return sparse domain at degree d. """
        return self.fibers[d] 

    def index (self, js, output=None):
        """ Index i of hyperedge [j0, ..., jn]. """
        if not self.trivial:
            return self.scalars().index(js, output)
        #--- Degree access
        if isinstance(js, int):
            return js
        #--- Index of hyperedge batch
        js = io.readTensor(js)
        n = 0
        while js.shape[-1] != (n + 1) + self.nlabels[n]:
            n += 1
        offset = self.sizes[n]
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

    def index_fmap(self, f):
        return torch.stack([self.index(f[0]), self.index(f[1])])

    def coords(self, i, d=None):
        """ Hyperedge [j0, ..., jn] at index i. """
        if not self.trivial:
            return self.scalars().coords(i, d)
        i, begin = io.readTensor(i), 0
        if not isinstance(d, type(None)):
            return self.grades[d][i]
        off = self.sizes.cumsum(0).contiguous()
        d = torch.bucketize(i, off, right=True)
        d = int(d.flatten()[0]) if d.numel() > 1 else int(d)
        if d >= 0 and d < len(self.grades):
            return self[d].keys[i - (off[d-1] if d > 0 else 0)]
        raise IndexError(f'Invalid index {i} for size {self.size} domain {self}')

    def coords_fmap(self, f):
        """ Map a pair of indices to coordinates. """
        return self.coords(f[0]).contiguous(), self.coords(f[1]).contiguous()

    def __len__(self):
        """ Maximal degree. """
        return len(self.fibers)

    def __repr__(self):
        return f"{self.dim} MultiGraph {self}"

    def show (self, json=False):
        s = '{\n\n'
        degree = lambda d: (f'  {d}: ' if not json else f'  "{d}": ')
        s += io.alignString(degree(0), self[0].keys) + ',\n\n'
        for d in range(1, self.dim + 1):
            s += io.alignString(degree(d), self[d].keys) \
                + (',\n\n' if d < self.dim else '\n\n')
        return s + '}'
        
