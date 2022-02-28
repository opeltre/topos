import torch
from topos.core import sparse, Field
from topos.io   import readTensor, readKey

class Domain: 
    
    def __init__(self, keys, shape=None, degree=None):
        """              """
        #--- Degree 
        self.degree = None
        #--- Keys + indices
        self.keys = keys
        self.idx = {readKey(k): i for i, k in enumerate(keys)}
        #--- Fiber shapes
        self.trivial = isinstance(shape, type(None))
        if self.trivial:
            self.shape = [[]] * len(keys)
        elif callable(shape):
            self.shape = [shape(a) for a in keys]
        #--- Pointers 
        sizes  = torch.tensor([readTensor(s).long().prod() for s in self.shape])
        offset = sizes.cumsum(0)
        begin  = torch.cat([torch.tensor([0]), offset])
        self.size  = offset[-1]
        self.sizes = sizes
        self.begin = begin[:-1]
        self.end   = begin[1:]
        #--- Cache
        self._cache = {}
    
    #--- Index range ---

    def __iter__(self):
        pass
    
    def items(self): 
        return self.idx.items()

    def index(self, key, *js): 
        """ Index of a key """
        if isinstance(key, int):
            return key
        else:
            return self.idx[key]

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
