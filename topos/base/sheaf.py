from .domain    import Domain
from .fiber     import Fiber

from topos.io   import readTensor, readFunctor 
from topos.core import sparse

import torch
import fp

class Sheaf (Domain):
    """
    Dictionary of domains.
    """

    @classmethod
    def sparse (cls, shape, indices, functor=None, degree=None):
        idx  = readTensor(indices, dtype=torch.long)
        adj  = sparse.tensor(shape, idx, dtype=torch.long).coalesce()
        sheaf = cls(adj, functor, degree)
        sheaf.adj = adj
        sheaf.is_sparse = True
        return sheaf

    def __init__(self, keys=None, functor=None, degree=None):
        """  
        Create sheaf from keys and domains. 

        If keys is an adjacency tensor then a sparse sheaf instance
        is created. 
        
        Examples:
        --------
        The following definitions are equivalent in each group:

            # Trivial sheaf on two elements (~ R^2)
            Sheaf({'a': [], 'b': []}]
            Sheaf(['a', 'b'])
            Sheaf(['a', 'b'], [None, None])
            Sheaf(['a', 'b'], functor=lambda _ : [])

            # Sheaf with distinct shapes n each key
            Sheaf({'a': [2, 3], 'b': [3, 4]})
            Sheaf(['a', 'b'], [[2, 3], [3, 4]])
        """
        is_domain = lambda f: 'coords' in dir(f) or isinstance(f, Domain)
        is_trivial = lambda f: f.trivial if 'trivial' in dir(f) else f.size == 1
        #--- Sparse sheaves ---
        self.is_sparse, self.adj = False, None
        #--- Fiber index ---
        self.idx, self.keys, fibers = readFunctor(keys, functor)
        self.fibers = [f if is_domain(f) else Fiber(None, f) for f in fibers]
        #--- Domain attributes ---
        self.trivial = all(is_trivial(f) for f in self.fibers)
        sizes  = torch.tensor([fiber.size for fiber in self.fibers])
        offset = sizes.cumsum(0)
        size = offset[-1]
        super().__init__(size, degree)
        #--- Pointers --- 
        begin  = torch.cat([torch.tensor([0]), offset])
        self.sizes = sizes
        self.begin = begin[:-1]
        self.end   = begin[1:]
   
    def index(self,  key, output=None):
        """ Index of a key """
        if self.is_sparse:
            idx = sparse.select(self.idx, key)
            if output == "mask":
                mask = sparse.index_mask(self.adj, key)
                return idx, mask
            return idx
        else:
            return self.idx[key]

    def arrow (self, a, b):
        n = self.size
        if self.is_sparse and self.trivial:
            ij = torch.stack([self.index(a), self.index(b)])
            return sparse.tensor([n, n], ij, t=False)
        print("arrow None")
        
    def slice(self, key):
        """ Return (begin, end, fiber) triplet at key. """
        idx = self.index(key)
        return self.begin[idx], self.end[idx], self.fibers[idx]

    def slices(self):
        """ Yield (begin, end, fiber) triplets. """
        for i,j, fiber in zip(self.begin, self.end, self.fibers):
            yield (i, j, fiber)

    def __getitem__(self, key):
        idx = self.index(key)
        return self.fibers[idx]

    def items(self):
        for k, fk in zip(self.keys, self.slices()):
            yield (k, fk)

    def __iter__(self):
        return self.keys.__iter__()

    def __repr__(self):
        return f"Sheaf {self}"
