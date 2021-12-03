from topos.base import Fiber
from topos.core import Field, Functional, Linear
from topos.core.operators import from_scalar, pullback, eye

import torch

class Domain: 
    """
    Base class for Sheaves. 

    Domains hold a dictionnary to Fiber objects, 
    which are pointers to an index range. 

    Subclasses should implement attributes: 

        .size   : int
        .degree : int   | None    
        .ftype  : Fiber | Simplex | ... 
        .fibers : {Fiber} 
    """
    
    def __init__(self, keys, shape=None, degree=None, ftype=Fiber):

        self.degree  = degree

        #--- Domain({"a": [3, 2], ...}) => sheaf ---
        if isinstance(keys, dict):
            shape = {ftype.read(k): Ek for k, Ek in keys.items()}
            keys  = list(shape.keys())
        #--- Domain(["a", "b", ...]) => trivial sheaf ---
        else:
            keys  = [ftype.read(k) for k in keys]

        #--- Join Fibers ---
        self.ftype  = ftype
        self.fibers, self.size = ftype.join(keys, shape)

    def get(self, key):
        """ Retrieve a fiber from its key. """
        key = self.ftype.read(key)
        return self.fibers[key]
    
    def __getitem__(self, key):
        """ Retrieve a fiber from its key. """
        if type(key) == int:
            return self
        return self.get(key)

    def __iter__(self):
        """ Yield fibers. """
        return self.fibers.values().__iter__()
    
    def keys (self):
        return self.fibers.keys()

    def items (self):
        return self.fibers.items()

    def shape(self, k):
        return self[k].shape.n

    #--- Index range ---

    def index(self, key, *js): 
        """ Get pointer to coordinate (j0, ..., jn) from fiber at key."""
        fiber = self.get(key)
        return fiber.begin + fiber.shape.index(*js)

    def range(self, d=None):
        """ Represent the domain mapped to its range of indices. """
        tgt = self if d == None else self[d]
        return tgt.field(torch.arange(tgt.size), d)
   
    #--- Subdomain --- 

    def restriction(self, keys):
        """ Domain restricted to a subset of keys. """
        if isinstance(keys, Domain):
            keys = keys.keys()
        keys  = [self.get(k).key for k in keys]
        shape = {k: self.get(k).shape for k in keys}
        return self.__class__(keys, shape, self.degree)

    #--- Functoriality ---

    def map(self, f, name="map \u033b"):
        """ Map a function acting on torch.Tensor to fields. """
        return Functional.map([self], f, name)

    def pull(self, src, g=None, name="map*", fmap=None):
        """ Pull-back of g : src -> domain. """
        mat = pullback(src, self, g, fmap)
        return Linear([self, src], mat, name)

    def push(self, tgt, f=None, name="map.", fmap=None):
        """ Push-forward of f : domain -> tgt. """
        mat = pullback(self, tgt, f, fmap).t()
        return Linear([self, tgt], mat, name)

    def res (self, keys, name="Res"):
        """ Restriction matrix to a subdomain. """
        tgt = keys if isinstance(keys, Domain)\
                   else self.restriction(keys)
        return self.pull(tgt, None, name)  
    
    def proj (self, keys, name="Proj"):
        """ Restriction projector. """
        res = self.res(keys)
        return (res.t() @ res).rename(name)

    def embed(self, subdomain, name="Emb"):
        """ Embedding matrix from a subdomain. """
        return self.res(subdomain).t().rename(name)

    def eye(self):
        """ Identity matrix."""
        return Linear([self], eye(self.size), "Id")

    #--- Field Creation ---

    def field(self, data, degree=None):
        """ Create a field from numerical data. """
        d = self.degree if degree == None else degree 
        return Field(self, data, d)

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

    def uniform(self, degree=None):
        """ Return uniform local probabilities. """
        tgt = self if degree == None else self[degree]
        return tgt.gibbs(self.zeros(degree))
    
    #--- Show --- 

    def __str__(self):
        return "{"  +\
               ", ".join([str(ck) for ck in self]) +\
               "}"
    
    def __repr__(self):
        return f"Domain {self}"
