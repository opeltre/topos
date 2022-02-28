import abc
import torch
from topos.core import sparse, Field
from topos.io   import readTensor, readFunctor

class Domain(abc.ABC):

    def __init__(self, size, degree=None):
        self.degree = degree
        self.size = size
        self._cache = {}

    #--- Field Creation ---

    def field(self, data, degree=None):
        """ Create a field from numerical data. """
        d = self.degree if degree == None else degree
        return Field(self, data, d)

    def from_scalars(self, x):
        if self.trivial:
            return x

    def zeros(self, degree=None):
        """ Return the unit of + field 0. """
        return self.field(0, degree)

    def ones(self, degree=None):
        """ Return the unit of * field 1. """
        return self.field(1, degree)

    def randn(self, degree=None):
        """ Return a field with normally distributed values. """
        tgt = self if degree == None else self[degree]
        return tgt.field(torch.randn(tgt.size), degree)

    def range(self, d=None):
        """ Represent the domain mapped to its range of indices. """
        tgt = self if d == None else self[d]
        return tgt.field(torch.arange(tgt.size), d)
    
    @abc.abstractmethod
    def slices(self):
        """ Yield (begin, end, domain) triplets."""
        yield (0, self.size, self)

    @abc.abstractmethod
    def slice(self, key=None):
        """ Slice (begin, end, domain) at key k. """
        return (0, self.size, self)

class Fiber (Domain):
    """
    Elementary domain.
    """

    def __init__(self, key=None, shape=None, degree=None):
        self.key = key
        self.trivial = isinstance(shape, type(None))
        if self.trivial:
            size  = 1
            self.shape = []
        else:
            shape = readTensor(shape)
            size  = shape.prod().long()
            self.shape = shape.tolist()
        #--- Domain attributes
        self.degree = degree
        self.size   = size
        super().__init__(size, degree)

    def slice(self, key=None):
        return (0, self.size, self)

    def slices(self):
        yield (0, self.size, self)

    def __getitem__(self, key=None):
        return self
  
    def __iter__(self):
        yield self

    def items(self):
        yield self.key, self


class Sheaf (Domain):
    """
    Dictionary of domains.
    """

    @classmethod
    def sparse (cls, shape, indices, functor=None, degree=None):
        idx  = readTensor(indices, dtype=torch.long)
        keys = sparse.tensor(shape, idx)
        return cls(keys, functor, degree)

    def __init__(self, keys=None, functor=None, degree=None):
        """              """
        #--- Fiber index ---
        self.idx, self.keys, fibers =\
            readFunctor(keys, functor)
        self.fibers = [fk if isinstance(fk, Domain) else Fiber(k, fk)\
                          for k, fk in zip(self.keys, fibers)]
        #--- Domain attributes ---
        self.trivial = all(f.trivial for f in self.fibers)
        sizes  = torch.tensor([fiber.size for fiber in self.fibers])
        offset = sizes.cumsum(0)
        size = offset[-1]
        super().__init__(size, degree)
        #--- Pointers --- 
        begin  = torch.cat([torch.tensor([0]), offset])
        self.sizes = sizes
        self.begin = begin[:-1]
        self.end   = begin[1:]
   
    def index(self, key):
        """ Index of a key """
        if isinstance(key, int):
            return key
        else:
            return self.idx[key]

    def slice(self, key):
        idx = self.index(key)
        return self.begin[idx], self.end[idx], self.fibers[idx]

    def slices(self):
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


class IndexSheaf(Sheaf):

    def __init__(self, keys, shape=None):
        #--- Adjacency tensor ---
        keys  = readTensor(keys, dtype=torch.long).T
        shape = [k.max() + 1 for k in keys]
        self.adj  = sparse.tensor(shape, keys, t=False).coalesce()
        self.keys = self.adj.indices().T
        #--- Index tensor ---
        idx = torch.arange(self.keys.shape[0])
        self.idx  = sparse.tensor(shape, self.keys.T, idx) 
        #--- Fibers ---
        self.trivial = isinstance(shape, type(None))
        if self.trivial: 
            shape = [[] * len(keys)]
        elif callable(shape):
            shape = [shape(k) for k in self.keys]
        self.fibers = [Fiber(k, s) for k, s in zip(self.keys, shape)]
        #--- Pointers ---
        self.sizes = torch.tensor([s.prod() for s in shape])
        offset = self.sizes.cumsum(0)
        self.begin = offset[:-1]
        self.end   = offset[1:]
        #--- Domain attributes
        size   = offset[-1]
        degree = keys.shape[-1]
        super(Sheaf, self).__init__(size, degree)
        super(Sheaf, self).__init__(size, degree)

        

