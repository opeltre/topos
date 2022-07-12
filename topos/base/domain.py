import abc
import torch

from topos.core import sparse, Field, Shape
from topos.io   import alignString

class Domain(abc.ABC):

    def __init__(self, size, degree=None, device=None):
        self.degree = degree
        self.size   = size
        self.device = device
        self._cache = {}

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
    
    @abc.abstractmethod
    def slices(self):
        """ Yield (begin, end, domain) triplets."""
        yield (0, self.size, self)

    @abc.abstractmethod
    def slice(self, key=None):
        """ Slice (begin, end, domain) at key k. """
        return (0, self.size, self)

    def __str__(self):
        return self.show()
    
    def __repr__(self):
        return "Domain {self}"

    def show (self, json=False):
        return f'(size:{self.size})'
