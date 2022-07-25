from .domain    import Domain
from .sheaf     import Sheaf

from .multigraph import MultiGraph

from topos.core import sparse
import topos.io as io

import fp
import torch

# Lexicographic sorting of (src, tgt) for SparseQuiver arrows
def lexsorting(a, b):
    """ Index map sorting integer tensors (a, b) lexicographically. """
    m = b.max() + 1
    x = m * a + b
    idx = x.sort().indices
    return idx


class Quiver(MultiGraph):
    """
    Quivers are sets of vertices and arrows, i.e. directed  1-multigraphs
    """

    def __init__(self, keys, arrows, functor=None, sort=True):
        """
        Create a quiver from integer tensors. 

        The last dimension of the arrows tensor is understood as [src, tgt, ...].
        This means shape [N, 2 + d] for d > 0 can be used to describe arrows 
        with their multiplicities.

        For fast lookup during `Q.hom(i, j)`, we cache two sparse matrices 
        `Q.begin_hom` and `Q.end_hom` yielding index ranges of the 
        `Q._arrows` tensor. 
        """
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
            graphF = [functor.fmap(a) for a in arrows]
            graphF = self[1].field(torch.cat(graphF))
        else:
            graphF = self[1].zeros()
        self.functor_graph = graphF
    
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

    def __repr__(self):
        return f"Quiver {self}"


class Diagram (Quiver):
    """
    Diagrams carry index maps between some of their fibers.

    The purpose of this class is to provide fast access to 
    (batched) functorial maps, returned as indices-value pairs. 
    """ 
    def __init__(self, quiver, functor=None):
        self.base = quiver
        self.keys = quiver.keys
        self.Obj = Sheaf(quiver[0], functor)
        self.Hom = Sheaf(quiver[1], lambda f : functor(quiver.source(f)))
        images = torch.cat([functor.fmap(a) for a in quiver.arrows()])        
        self.images = self.Hom.field(images)

    def hom(self, src, tgt):
        """ Return arrows between source and target. """
        f = self.base.hom(src, tgt) 

    def fmap(self, a):
        pass

    def __repr__(self):
        return f'Diagram {self.base}'


class Functor:

    def __init__(self, f0, f1):
        self.obj_map = f0
        self.hom_map = f1

    def __call__(self, a):
        return self.obj_map(a)
    
    def fmap(self, f):
        return self.hom_map(f)

class FreeFunctor:

    def __init__(self, shape):
        if not callable(shape)  : shape = shape.__getitem__
        def obj(a)  : return fp.Torus([shape(i) for i in a])
        def fmap(f) : return obj(f.src).res(*f.indices)
        super().__init__(obj, fmap)

class Supset(fp.Arrow):

    def __new__(cls, A, B):
        TAB = super().__new__(cls, A, B)
        def _init_(Arr, indices):
            f.indices = indices
        return 

    def __init__(self, A, B):
        super().__init__(A, B)
        self.arity = 0