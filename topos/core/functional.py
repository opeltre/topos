from .system import System
from .field import Field
import torch

class Functional :

    def __init__(self, f, degree=0, name="\u033b"):
        self.call = f
        self.degree = degree
        self.name = name

    def __call__(self, field):
        data = self.call(field.data)
        return Field(field.system, field.degree + self.degree, data)

    def __matmul__(self, other): 
        if isinstance(other, Field):
            return self(other)
        if isinstance(other, Functional):
            degree = self.degree + other.degree
            f = lambda d: self.call(other.call(d))
            name = f"{self.name} . {other.name}"
            return self.__class__(f, degree, name)
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.degree} Functional {self}"

class Matrix (Functional): 
    
    def __init__(self, mat, degree=0, name="Mat"):
        self.mat = mat
        f = lambda d: mat @ d
        super().__init__(f, degree, name)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            degree = self.degree + other.degree
            matmul = torch.sparse.mm
            mat = matmul(self.mat, other.mat)
            return Matrix(mat, degree)
        return super().__matmul__(other)

    def __repr__(self):
        return f"{self.degree} Linear {self}"
