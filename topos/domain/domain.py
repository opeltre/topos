from topos.base import join_cells, Chain, Shape, Cell
from topos.core import Field, Functional, Linear
from topos.core.operators import from_scalar, pullback, eye

import torch

class Domain : 
    """
    Base class for Sheaves. 

    Domains hold a dictionnary of `Fiber` objects,
    that are pointers to index ranges. 
    """

    def __init__(self, cells, degree=0, size=None):
        self.degree = degree
        self.cells  = cells
        self.size = size if size else\
                    max((c.end for c in cells.values())) 

    def __iter__(self):
        """ Yield cells. """
        return self.cells.values().__iter__()

    def get(self, key):
        """ Retrieve a cell from its key. """
        return self.cells[key]

    def __getitem__(self, key):
        """ Retrieve a cell from its key. """
        if type(key) == int:
            return self
        return self.get(key)

    def index(self, key, *js): 
        """ Get pointer to coordinate (j0, ..., jn) from cell at key."""
        cell = self[key]
        return cell.begin + cell.shape.index(*js)
    
    #--- Restricted domain to a subset of keys --- 

    def restriction(self, keys):
        keys  = [self.get(k).key for k in keys]
        shape = {k: self.get(k).shape for k in keys}
        return self.__class__(keys, shape, self.degree)

    #--- Functors ---

    def map(self, f, name="map \u033b"):
        """ Map a function acting on torch.Tensor to fields. """
        return Functional.map([self], f, name)

    def pull(self, src, g=None, name="map*"):
        """ Pull-back of g : src -> domain. """
        mat = pullback(src, self, g)
        return Linear([self, src], mat, name)

    def push(self, tgt, f=None, name="map."):
        """ Push-forward of f : domain -> tgt. """
        mat = pullback(self, tgt, f).t()
        return Linear([self, tgt], mat, name)

    def res (self, keys, name="Res"):
        subdomain = keys if isinstance(keys, Domain)\
                    else self.restriction(keys)
        return self.pull(subdomain, None, name)  

    def embed(self, subdomain, name="Emb"):
        return self.res(subdomain).t().rename(name)

    def eye(self):
        return Linear([self], eye(self.size), "Id")

    #--- Field Creation ---

    def field(self, data, degree=0):
        """ Create a field from data vector. """
        return Field(self, data, self.degree)

    def zeros(self, degree=0):
        """ Return the unit of + field 0. """
        return self.field(0, degree)

    def ones(self, degree=0):
        """ Return the unit of * field 1. """
        return self.field(1, degree)

    def randn(self, degree=0):
        """ Return a field with normally distributed values. """
        return self.field(torch.randn(self[degree].size), degree)

    def uniform(self, degree=0):
        """ Return uniform local probabilities. """
        return self.gibbs(self.zeros(degree))
    
    #--- Show --- 

    def __str__(self):
        return "{"  +\
               ", ".join([str(ck) for ck in self]) +\
               "}"

    def __repr__(self):
        return f"Domain {self}"


