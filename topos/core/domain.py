from topos.base import join_cells, Chain, Shape, Cell
from .vect import Vect
from .field import Field
from .functional import Functional
from .linear import Linear
from .operators import from_scalar, pullback, eye
import torch

matmul = torch.sparse.mm

class Domain : 

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

    #--- Functors ---

    def map(self, f, name="map \u033b"):
        """ Map a function acting on torch.Tensor to fields. """
        return Functional.map([self], f, name)

    def pull(self, src, g, name="map*"):
        """ Pull-back of g from src to domain. """
        mat = pullback(src, self, g)
        return Linear([self, src], mat, name)

    def push(self, src, f, name="map."):
        mat = pull(src, self, g).t()
        return Linear([src, self], mat, name)

    def restrict(self, subdomain, name="Res"):
        if not isinstance(subdomain, Domain):
            shape = {a: self[a].shape for a in self.cells} 
            keys = [self[k].key for k in subdomain]
            subdomain = Sheaf(keys, shape, self.degree)
        def incl (cb):
            return cb.key
        return self.pull(subdomain, incl, name)

    def embed(self, subdomain, name="Emb"):
        return self.restrict(subdomain).t().rename(name)

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


class Sheaf (Domain) : 
    def __init__(self, keys, shape=None, degree=0):
        """
        Create a domain with hashable keys and given shapes.

        Domains hold a dictionnary of Cell objects,
        that are essentially pointers to index ranges 
        of size `n1 * ... * nd` for shapes `(n1, ..., nd)`. 

        Params: - keys  : [Hashable]
        ------  - shape : {keys : Shape} | keys -> Shape

        """

        self.trivial = (shape == None)
        self.scalars = self.__class__(keys) if not self.trivial else self

        #--- Join Cells ---
        if shape == None:
            shape = {k : Shape() for k in keys}
        elif callable(shape):
            shape = {k: shape(k) for k in keys}
        cells, size = join_cells(keys, shape)

        super().__init__(cells, degree, size=size)

        
        #--- From/To scalar fields ---
        src, J = self.scalars, from_scalar(self)
        extend = Linear([src, self], J, "J")
        sums   = Linear([self, src], J.t(), "\u03a3")
        #--- Normalisation ---
        norm    = Functional([self], lambda f: f/sums(f), "(1 / \u03a3)")
        #--- Energies / log-likelihoods ---
        _ln     = self.map(lambda d: -torch.log(d), "(-ln)")
        #--- Gibbs states / densities ---
        exp_    = self.map(lambda d: torch.exp(-d), "(e-)")
        gibbs   = (norm @ exp_).rename("(e- / \u03a3 e-)")
        
        self.maps = {
            "extend"    : extend,
            "sums"      : sums  ,
            "normalise" : norm  ,
            "_ln"       : _ln   ,   
            "exp_"      : exp_  ,
            "gibbs"     : gibbs      
        }
        for k, fk in self.maps.items():
            setattr(self, k, fk)

    def __repr__(self):
        return f"Sheaf  {self}" if not self.trivial else f"Domain {self}"

#--- System Fibers ---

class Simplicial (Sheaf) : 
    """ Domain with simplicial Chain keys """

    def __init__(self, keys, shape=None, degree=0):
        chains = [Chain.read(k) for k in keys]
        shape  = {Chain.read(k): shape(k) for k in keys} if shape\
                else None
        super().__init__(chains, shape, degree)
    
    def get(self, key):
        return super().get(Chain.read(key))

    def __repr__(self):
        return f"{self.degree} {super().__repr__()}"


#--- Scalar Domain --- 

class Trivial (Domain) :
    """ Domain with point fibers. """

    def __init__(self, keys, degree=0):
        self.degree = degree
        self.shape = {k: Shape() for k in keys}
        self.cells, self.size = join_cells(keys, self.shape)

#--- Unit Object ---

class Point (Domain): 
    """ Point Domain spanning field of scalars R. """

    def __init__(self, degree=0):
        super().__init__({'()': Cell('()', 0, Shape())}, degree)


#--- Null Object ---

class Empty (Point):
    """ Empty Domain spanning the null vector space {0}. """

    def field(self, data=None):
        return super().field(torch.tensor([0.]), self.degree)
