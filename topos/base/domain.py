import abc
import torch
from topos.core import sparse, Field
from topos.io   import readTensor, readKey


class Domain(abc.ABC):
   
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
        pass

    @abc.abstractmethod
    def slice(self, key):
        """ Slice (begin, end, domain) at key k. """
        pass

class Fiber (Domain):
    """
    Elementary domain.
    """

    def __init__(self, key=None, shape=None, degree=None):
        self.key = key
        self.degree = None
        if isinstance(shape, type(None)):
            self.shape = []
            self.size  = 1
        else:
            shape = readTensor(shape)
            self.shape = shape.tolist()
            self.size  = shape.prod().long()

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

    def __init__(self, keys, shape=None, degree=None):
        """              """
        #--- Degree
        self.degree = None
        #--- Keys + indices
        self.keys = keys
        self.idx = {readKey(k): i for i, k in enumerate(keys)}
        #--- Fibers 
        self.trivial = isinstance(shape, type(None))
        if self.trivial:
            shape = [[]] * len(keys)
        elif callable(shape):
            shape = [shape(a) for a in keys]
        self.fibers = [fk if isinstance(fk, Domain) else Fiber(k, fk)\
                          for k, fk in zip(keys, shape)]
        #--- Pointers
        sizes  = torch.tensor([fiber.size for fiber in self.fibers])
        offset = sizes.cumsum(0)
        begin  = torch.cat([torch.tensor([0]), offset])
        self.size  = offset[-1]
        self.sizes = sizes
        self.begin = begin[:-1]
        self.end   = begin[1:]
        #--- Cache
        self._cache = {}
   
    def index(self, key, *js):
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
