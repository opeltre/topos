from .functional import Functional, GradedFunctional
from .vect import Vect
from .sparse import eye, zero

import torch

class Linear (Functional, Vect): 
    
    def __init__(self, domains, mat=0, name="Mat"):

        src, tgt = domains[0], domains[-1]
        
        #-- Zero matrix --
        if isinstance(mat, int) and mat == 0:
            name = "0"
            mat  = zero(tgt.size, src.size)
        #-- Scalars * Identity --
        if not isinstance(mat, torch.Tensor) and tgt == src:
            name = str(mat)
            mat  = mat * eye(tgt.size)

        #-- Action on fields
        def matvec(x):
            return tgt.field(mat @ x.data)

        self.data = mat
        super().__init__([src, tgt], matvec, name)
    
    @classmethod
    def null(cls, domains):
        return cls(domains, 0, "0")

    def same(self, data=None): 
        if isinstance(data, type(None)):
            data = self.data
            name = self.name
        elif isinstance(data, torch.Tensor):
            name = "Mat"
        elif self.data.shape[0] == self.data.shape[1]:
            name = str(data)
            data = data * eye(self.data.shape[0])
        return Linear(data, self.degree, name)
    
    def compose(self, other):
        tgt, src = self.tgt, other.src
        matmul = torch.sparse.mm
        mat = matmul(self.data, other.data)
        return Linear([src, tgt], mat, f"{self} . {other}")

    def __matmul__(self, other):
        if isinstance(other, Linear):
            return self.compose(other)
        return super().__matmul__(other)

    def __truediv__(self, other): 
        return self.same(self.data / other)

    def __repr__(self):
        return f"Linear {self}"

    def __neg__(self):
        return super().__neg__().rename(f"- {self.name}")

    def t(self):
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
    
    def __init__(self, Ks, mats, degree=0, name="Mat"):
        """
        Create a graded operator from an array of matrices.
        """
        super().__init__(Ks, mats, degree, name, cls=Linear)
    
    def null(self, i):
        return Linear([self.src[i], self.tgt[i + self.degree]])

    #--- Show ---

    def __repr__(self):
        return f"{self.degree} Linear {self}"
