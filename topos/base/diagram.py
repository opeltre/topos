from .domain import Domain
from topos.core   import sparse

import fp
import torch

def lexsorting(a, b):
    """ Index map sorting integer tensors (a, b) lexicographically. """
    m = b.max() + 1
    x = m * a + b
    idx = x.sort().indices
    return idx

class Category(Domain):
    """ 
    Small (finite) categories.

    Objects represent domain keys e.g. indices rather than types 
    (although small categories could be embedded in the category of types).
    For objects occupying more than one index place, use the Diagram
    class instead.

    Sparse categories store `torch.LongTensor` instances instead of
    dictionnaries to provide efficient batched access to arrows. 
    """
    def __init__(self, keys, arrows, device=None, sort=False):
        """
        Create a category from iterable of keys and arrows. 
        
        If `arrows` is a 2D-tensor, then the first 2 elements of the 
        last dimension will be understood as source and target.  
        For sparse categories, two `.begin_hom` and `.end_hom` 
        sparse matrices are computed, yielding index ranges 
        of the `arrows` tensor. 

        For generic categories, a `._hom` dictionnary of lists
        is computed, yielding arrows between a given source and target.
        """
        super().__init__(len(keys), degree=0, device=None)
        self._keys   = keys
        self._arrows = arrows
        self.is_sparse = (isinstance(keys, torch.LongTensor) and
                          isinstance(arrows, torch.LongTensor))
        
        #--- dictionnary of arrows
        if not self.is_sparse:
            self._hom = {}
            for f in self._arrows:
                a, b = f.src, f.tgt if isinstance(f, fp.Arrow) else f
                if not (a, b) in self._hom:
                    self._hom[(a, b)] = []
                self._hom[(a, b)].append(f)

        #--- sparse arrow indexing
        if self.is_sparse:
            idx_hom = torch.arange(arrows.shape[0])
            src = arrows[:,0]
            tgt = arrows[:,1]
            # sort (src, tgt) lexicographically
            if sort: 
                s = lexsorting(src, tgt)
                idx_hom, src, tgt = idx_hom[s], src[s], tgt[s]
            ij = torch.stack([src, tgt])
            # find (src, tgt) dicontinuities
            N, n = ij.shape[-1], self.size
            dij  = torch.diff(ij).float().norm(dim=[0])
            nz   = dij.nonzero().flatten()
            b = torch.cat([torch.tensor([0]), nz])
            e = torch.cat([nz, torch.tensor([N - 1])])
            # store arrow index ranges
            begin = sparse.tensor([n, n], ij.index_select(1, b), idx_hom[b], t=0)
            end   = sparse.tensor([n, n], ij.index_select(1, e), idx_hom[e], t=0)
            self.begin_hom = begin
            self.end_hom   = end
        
        self.Hom = Domain(len(arrows), degree=1, device=device)

    def hom(self, src, tgt):
        """ Return arrows between source and target. """
        if not self.is_sparse:
            return self._hom[(src, tgt)]
        ij = torch.stack([src, tgt])
        begin = sparse.select(self.begin_hom, ij)
        end   = sparse.select(self.end_hom, ij)
        return self._arrows[begin:end]
    
    def arrows(self):
        """ Yield arrows """
        if not self.is_sparse:
            return self._arrows.__iter__()


class Diagram (Category):
    """
    Diagrams carry index maps between some of their fibers.

    The purpose of this class is to provide fast access to 
    (batched) functorial maps, returned as indices-value pairs. 
    """ 
    def __init__(self, cat, functor=None):
        self._keys   = cat._keys
        self.fibers  = [functor(k) for k in cat.keys]
        self._arrows = [functor.fmap(f) for f in cat.arrows()]

    def hom(self, src, tgt):
        """ Return arrows between source and target. """
        f = self.cat.hom(src, tgt) 

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