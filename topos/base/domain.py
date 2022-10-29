import abc
import torch

from topos.core import sparse, Field, Shape
from topos.io   import alignString

import fp

class Domain(fp.meta.Type):

    def __init__(self, size, degree=None, device=None):
        self.degree = degree
        self.size   = size
        self.shape  = [size]
        self.device = device
        self._cache = {}

    def torch_fmap(self, f, name='\u033b'):
        def map_f(x):
            if isinstance(x, int):
                src = self.Field(x)
                F = fp.Arrow(src, src)(f)
                F.__name__ = name
                return F
            elif isinstance(x, Field):
                src = self.Field(x.degree)
                return src(f(x))
        map_f.__name__ = name
        return map_f

    #--- Field Creation ---

    def Field(self, degree=None):
        """ Field class (optionally restricted to a graded component). """
        tgt = self if degree == None else self[degree]
        return Field(tgt)
        
    def field(self, data, degree=None):
        """ Create a field from numerical data. """
        return self.Field(degree)(data)

    def from_scalars(self, x):
        if self.trivial:
            return x

    def zeros(self, degree=None):
        """ Return the unit of + field 0. """
        return self.Field(degree).zeros()

    def ones(self, degree=None):
        """ Return the unit of * field 1. """
        return self.Field(degree).ones()

    def randn(self, degree=None):
        """ Return a field with normally distributed values. """
        tgt = self if degree == None else self[degree]
        return self.field(torch.randn(tgt.size), degree)

    def range(self, d=None):
        """ Represent the domain mapped to its range of indices. """
        tgt = self if d == None else self[d]
        return self.field(torch.arange(tgt.size), d)

    #--- Fiber iteration --- 

    def __iter__(self):
        return range(self.size).__iter__()

    def slices(self):
        """ Yield (begin, end, domain) triplets."""
        yield (0, self.size, self)

    def slice(self, key=None):
        """ Slice (begin, end, domain) at key k. """
        return (0, self.size, self)

    #--- Show --- 

    def __str__(self):
        return self.show()
    
    def __repr__(self):
        return f"Domain {self}"

    def show (self, json=False):
        return (self.__name__ if '__name__' in dir(self)
                              else f'(size:{self.size})')
