from .functional import Functional
from .vect import Vect

import torch

class Matrix (Functional, Vect): 
    
    def __init__(self, mat, degree=0, name="Mat"):
        self.data = mat
        f = lambda d: mat @ d
        super().__init__(f, degree, name)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            degree = self.degree + other.degree
            matmul = torch.sparse.mm
            mat = matmul(self.data, other.data)
            return Matrix(mat, degree, f"{self} . {other}")
        return super().__matmul__(other)

    def __repr__(self):
        return f"{self.degree} Linear {self}"
