from .functional import Functional, GradedFunctional
from .vect import Vect
from .sparse import eye, zero, matmul, diag

import torch

class Linear (Functional, Vect): 
    """ 
    Linear functionals implemented as sparse matrices.
    """
    
    def __init__(self, domains, mat=0, name="Mat", degree=None):
        """
        Create a Linear operator between domains = [src, tgt].
        """
        src, tgt = domains[0], domains[-1]
        self.degree = degree
        
        #-- Null --
        if isinstance(mat, int) and mat == 0:
            name = "0"
            mat  = zero(tgt.size, src.size)

        #-- Identity --
        if not isinstance(mat, torch.Tensor) and tgt == src:
            name = str(mat)
            mat  = mat * eye(tgt.size)

        #-- Action on fields
        def matvec(x):
            if torch.is_complex(mat) and not torch.is_complex(x.data):
                return tgt.field(mat @ x.data.cfloat())
            return tgt.field(mat @ x.data)

        self.data = mat
        super().__init__([src, tgt], matvec, name)

    @classmethod
    def null(cls, domains):
        """
        Return the null operator between domains.
        """
        return cls(domains, 0, "0")

    def same(self, data=None, name=None): 
        """
        Return an operator between same domains from a matrix. 
        """
        src, tgt = self.src, self.tgt
        if isinstance(data, type(None)):
            data = self.data
            name = self.name
        elif isinstance(data, torch.Tensor):
            name = name if name != None else self.name 
        return Linear([src, tgt], data, name, self.degree)
    
    def compose(self, other):
        """
        Return the composed operator, multiplying matrices.
        """
        tgt, src = self.tgt, other.src
        mat = matmul(self.data, other.data)
        return Linear([src, tgt], mat, f"{self} . {other}")

    def __matmul__(self, other):
        """
        Composition of operators/ Action on fields.
        """
        if isinstance(other, Linear):
            return self.compose(other)
        elif isinstance(other, Vect):
            return self(other)
        return super().compose(other)

    def __truediv__(self, other): 
        """ 
        Divide coefficients by numerical data (e.g. scalar).
        """
        return self.same(self.data / other)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.src.field(self.data[i].to_dense())
        elif isinstance(i, str):
            return self[self.tgt.index(i)]
   
    def __mul__(self, other):
        if not isinstance(other, Linear):
            D = other.domain
            w = diag(D.size, other.data)
            if D == self.src:
                out = matmul(self.data, w)
            return self.same(out)
        return super().__mul__(other)

    def __rmul__(self, other):
        if not isinstance(other, Linear):
            D = other.domain
            w = diag(D.size, other.data)
            if D == self.tgt:
                out = matmul(w, self.data)
            return self.same(out)
        return super().__rmul__(other)
            


    def __repr__(self):
        return f"Linear {self}"

    def t(self):
        """ 
        Transposed operator.
        """
        return self.__class__(
            [self.tgt, self.src], 
            self.data.t(),
            name = f"{self}*")


class GradedLinear (GradedFunctional):
    """
    Graded Linear operators.

    An array `L = [L0, ..., Ld]` of linear operators 
    acting on a k-field `f` by:

        L(f) = L @ f = L[k] @ f
        
    """
    
    def __init__(self, domains, mats, degree=0, name="Mat"):
        """
        Create a graded operator from an array of matrices.
        """
        super().__init__(domains, mats, degree, name, cls=Linear)
    
    def null(self, i):
        return Linear([self.src[i], self.tgt[i + self.degree]])

    def t(self): 
        d    = self.degree
        name = f'{self}*'
        ts   = [self.null(i).t() for i in range(d)]\
             + [fi.t() for fi in self.grades]
        return self.__class__([self.tgt, self.src], ts, -d, name) 

    #--- Show ---

    def __repr__(self):
        return f"{self.degree} Linear {self}"
