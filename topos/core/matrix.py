from .functional import Functional
from .vect import Vect
from .sparse import eye

import torch

class Matrix (Functional, Vect): 
    
    def __init__(self, mat, degree=0, name="Mat"):
        self.data = mat
        def matmul(x):
            target = x.system[x.degree + degree] if degree else x.domain
            return target.field(mat @ x.data)
        super().__init__(matmul, degree, name)

    def same(self, data=None): 
        if isinstance(data, type(None)):
            data = self.data
            name = self.name
        elif isinstance(data, torch.Tensor):
            name = "Mat"
        elif self.data.shape[0] == self.data.shape[1]:
            name = str(data)
            data = data * eye(self.data.shape[0])
        return Matrix(data, self.degree, name)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            degree = self.degree + other.degree
            matmul = torch.sparse.mm
            mat = matmul(self.data, other.data)
            return Matrix(mat, degree, f"{self} . {other}")
        return super().__matmul__(other)

    def __truediv__(self, other): 
        return self.same(self.data / other)

    def __repr__(self):
        return f"{self.degree} Linear {self}"
