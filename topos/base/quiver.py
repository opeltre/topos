from .domain    import Domain
from .sheaf     import Sheaf

from .functor import Functor
from .multigraph import MultiGraph

from topos.core import sparse
import topos.io as io

import fp
import torch

# Lexicographic sorting of (src, tgt) for Quiver arrows
def lexsorting(a, b):
    """ Index map sorting integer tensors (a, b) lexicographically. """
    m = b.max() + 1
    x = m * a + b
    idx = x.sort().indices
    return idx


class Quiver(MultiGraph, Functor):
    """
    Quivers are sets of vertices and arrows, i.e. directed  1-multigraphs.

    Quiver diagrams carry index maps between some of their fibers.

    One purpose of this class is to provide fast access to 
    (batched) functorial maps, returned as indices-value pairs.
    """

    def __init__(self, grades, functor=None, sort=True):
        """
        Create a quiver from integer tensors. 

        The `grades = [keys, arrows]` argument should be a list of tensorlike
        objects representing vertices and relationships.

        The last dimension of the arrows tensor is understood as [src, tgt, ...].
        This means shape [N, 2 + d] for d > 0 can be used to describe arrows 
        with their multiplicities.

        For fast lookup during `Q.hom(i, j)`, we cache two sparse matrices 
        `Q.begin_hom` and `Q.end_hom` yielding index ranges of the 
        `Q._arrows` tensor. 
        """
        keys, arrows = (grades if not isinstance(grades, MultiGraph)
                               else grades.grades)

        self.is_sparse = True
        self.keys = keys
        
        idx_hom = torch.arange(arrows.shape[0])
        src = arrows[:,0]
        tgt = arrows[:,1]

        # sort (src, tgt) lexicographically
        if sort: 
            s = lexsorting(src, tgt)
            src, tgt = src[s], tgt[s]
            arrows = arrows[s]
        ij = torch.stack([src, tgt])
        
        # Obj and Hom domains in degrees 0, 1
        super().__init__([keys, arrows], functor)
        
        # find (src, tgt) dicontinuities
        N, n = ij.shape[-1], self.Nvtx
        dij  = torch.diff(ij).float().norm(dim=[0])
        nz   = dij.nonzero().flatten()
        b = torch.cat([torch.tensor([0]), 1 + nz])
        e = torch.cat([1 + nz, torch.tensor([N - 1])])

        # store arrow index ranges
        begin = sparse.tensor([n, n], ij.index_select(1, b), idx_hom[b], t=0)
        end   = sparse.tensor([n, n], ij.index_select(1, b), idx_hom[e], t=0)
        self.begin_hom = begin
        self.end_hom   = end

        # functor graph
        self.functor = functor
        if not isinstance(functor, type(None)):
            graphT = []
            for f in arrows:
                Tf = functor.fmap(f)
                src = functor(f[0])
                N = src.size if 'size' in dir(src) else fp.Torus(src).size
                gf = Tf(torch.arange(N)) 
                graphT.append(gf.data)
            graphT = self[1].field(torch.cat(graphT))
        else:
            graphT = self[1].zeros()
        self.functor_graph = graphT

    def source(self, a):
        """ Source of arrow. """
        a = io.readTensor(a).long()
        arr = self.grades[1][a]
        return arr[:,0] if a.dim() else arr[0]

    def target(self, a):
        """ Target of arrow. """
        a = io.readTensor(a).long()
        arr = self.grades[1][a]
        return arr[:,1] if a.dim() else arr[1]    
       
    def hom(self, src, tgt):
        """ Return arrows between src and tgt. """
        src, tgt = io.readTensor(src), io.readTensor(tgt)
        ij = torch.stack([src, tgt])
        begin = sparse.select(self.begin_hom, ij)
        end   = sparse.select(self.end_hom, ij)
        return self.grades[1][begin:end]

    def arrows(self):
        """ Yield arrows """
        return self.functor_graph 

    def fmap (self, f):
        """
        Functorial graph associated to f = [a, b] or f = [[*as], [*bs]].

        In the case of multiple arrows we return the concatenated 
        functorial graphs (i.e. coproduct of arrows). The associated
        sparse matrix acting on Q[0] will then yield the sum of arrows.
        """
        i, j = io.readTensor(f[0]), io.readTensor(f[1])
        i, j = i.view([-1]), j.view([-1])
        ij   = torch.stack([i, j])

        #--- Scalar coefficients ---
        if self.trivial:
            return ij

        #--- Functorial coefficients ---
        obj, hom = self[0], self[1]
        # object indices
        I = i.repeat_interleave(obj.sizes[i])
        J = j.repeat_interleave(obj.sizes[i])
        # arrow indices
        narr = hom.keys.shape[-1]
        f = io.readTensor(f, dtype=torch.long).view([narr, -1])
        k = hom.index(f.T)
        K = k.repeat_interleave(obj.sizes[i])
        # functor graph
        F = self.arrows().data
        # relative source index
        off_i  = obj.sizes[i].cumsum(0).roll(1)
        off_i[0] = 0
        off_I = off_i.repeat_interleave(obj.sizes[i])
        Ioff = torch.arange(I.shape[0]) - off_I
        # relative target index
        Joff = F[hom.begin[K] + Ioff]
        # absolute source and target indices
        Fi = obj.begin[I] + Ioff
        Fj = obj.begin[J] + Joff
        Fij = torch.stack([Fi, Fj])
        return Fij

    def __repr__(self):
        name = self.__name__ if '__name__' in dir(self) else 'Q'
        return name