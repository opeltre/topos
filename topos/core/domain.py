from topos.base import join_cells, Chain, Shape
from .vect import Vect
from .field import Field
from .functional import Functional
from .linear import Linear
from .operators import from_scalar, pull
import torch

matmul = torch.sparse.mm

class Domain : 

    def __init__(self, keys, shape, degree=0):
        """
        Create a domain with hashable keys and given shapes.

        Domains hold a dictionnary of Cell objects,
        that are essentially pointers to index ranges 
        of size `n1 * ... * nd` for shapes `(n1, ..., nd)`. 

        Params: - keys  : [Hashable]
        ------  - shape : {keys : Shape} | keys -> Shape

        """
        self.degree = degree

        #--- Join Cells ---
        if shape == None:
            shape = {k : Shape() for k in keys}
        elif callable(shape):
            shape = {k: shape(k) for k in keys}
        self.shape = shape
        self.cells, self.size = join_cells(keys, shape)

        #--- From/To scalar fields ---
        extend = from_scalar(self)
        sums = Linear([self], matmul(extend, extend.t()))
        def normalise(field):
            return field / sums(field)
        self.normalise = Functional([self], normalise, "(1 / \u03a3)")

        #--- Gibbs states and energy ---
        def _ln (data):
            return - torch.log(data)
        def exp_ (data):
            return torch.exp(-data)
        self._ln = self.map(_ln, "(-ln)")
        self.exp_ = self.map(exp_, "(e-)")
        self.gibbs = (self.normalise @ self.exp_)\
                     .rename("(e- / \u03a3 e-)")

    #--- Local pointers ---

    def __iter__(self):
        """ Yield cells. """
        return self.cells.values().__iter__()

    def __getitem__(self, key):
        """ Retrieve a cell from its key. """
        return self.cells[key]

    def index(self, key, *js): 
        """ Get pointer to coordinate (j0, ..., jn) from cell at key."""
        cell = self[key]
        return cell.begin + cell.shape.index(*js)

    #--- Functors ---

    def map(self, f, name="map \u033b"):
        """ Map a function acting on torch.Tensor to fields. """
        return Functional.map([self], f)

    def pull(self, src, g, name="map*"):
        """ Pull-back of g from src to domain. """
        mat = pull(src, self, g)
        return Linear([self, src], mat, name)

    def push(self, src, f, name="map."):
        mat = pull(src, self, g).t()
        return Linear([src, self], mat, name)

    def restrict(self, subdomain):
        if not isinstance(subdomain, Domain):
            shape = {a: self[a].shape for a in self.cells} 
            keys = [self[k].key for k in subdomain]
            subdomain = Domain(keys, shape, self.degree)
        def incl (cb):
            return cb.key
        return self.pull(subdomain, incl)

    def embed(self, subdomain):
        return self.restrict(self, subdomain).t()

    #--- Field Creation ---

    def field(self, data):
        """ Create a field from data vector. """
        return Field(self, data, self.degree)

    def zeros(self):
        """ Return the unit of + field 0. """
        return self.field(torch.zeros(self.size))

    def ones(self):
        """ Return the unit of * field 1. """
        return self.field(torch.ones(self.size))

    def randn(self):
        """ Return a field with normally distributed values. """
        return self.field(torch.randn(self.size))

    def uniform(self):
        """ Return uniform local probabilities. """
        return self.gibbs(self.zeros())
    
    #--- Show --- 

    def __str__(self):
        return "{"  +\
               ", ".join([str(ck) for ck in self]) +\
               "}"

    def __repr__(self):
        return f"Domain {self}"

class GradedDomain (Domain):

    def __init__(self, keys, shape, degree=0):
        self.degree = degree
        shape = {Chain.read(k): shape(k) for k in keys}
        super().__init__(shape.keys(), shape, degree)

    def __getitem__(self, key): 
        return super().__getitem__(Chain.read(key))

class EmptyDomain (Domain):

    def __init__(self):
        super().__init__([''], shape={'': Shape()})

    def field(self, data):
        return Vect(self, torch.tensor([0.]))
        
