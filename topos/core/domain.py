from topos.base import join_cells, Chain
from .vect import Vect
from .field import Field
from .functional import Functional
from .matrix import Matrix
from .operators import from_scalar, restrict
import torch

matmul = torch.sparse.mm

class Domain : 

    def __init__(self, keys, shape):

        if callable(shape):
            shape = {k: shape(k) for k in keys}

        #--- Pointers to local data ---
        self.cells, self.size = join_cells(keys, shape)

        #--- From/To scalar fields ---
        extend = from_scalar(self)
        sigma = extend.t()

        #--- Normalisation ---
        j_sigma = Matrix(matmul(extend, sigma), 0)
        def normalise(field):
            return field / j_sigma(field)
        self.normalise = Functional(normalise, 0, "(1 / \u03a3)")

        #--- Gibbs states ---
        def exp_ (data):
            return torch.exp(-data)
        self.exp_ = self.map(exp_, "(e-)")
        self.gibbs = self.normalise @ self.exp_
        self.gibbs.rename("(e- / \u03a3 e-)")

        #--- Energy --- 
        def _ln (data):
            return - torch.log(data)
        self._ln = self.map(_ln, "(-ln)")
    
    #--- Cells ---

    def __iter__(self):
        return self.cells.values().__iter__()

    def __getitem__(self, key):
        return self.cells[key]

    def index(self, a, *js): 
        cell = self[a]
        return cell.begin + cell.shape.index(*js)

    #--- Field Creation ---

    def field(self, data):
        return Vect(self, data)

    def zeros(self):
        return self.field(torch.zeros(self.size))

    def ones(self):
        return self.field(torch.ones(self.size))

    def randn(self):
        return self.field(torch.randn(self.size))
    
    #--- Functors ---


    def map(self, f, name="map \u033b"):
        return Functional.map(f, 0, name)

    #--- Show --- 

    def __str__(self):
        return "{"  +\
               ", ".join([str(ck) for ck in self]) +\
               "}"

    def __repr__(self):
        return "Domain"


class GradedDomain (Domain):

    def __init__(self, complex, degree, keys, shapes):
        self.degree = degree
        self.complex = complex
        super().__init__(keys, shapes)

    def field(self, data):
        return Field(self.complex, self.degree, data)
    
    def __getitem__(self, key):
        return self.cells[Chain.read(key)]
