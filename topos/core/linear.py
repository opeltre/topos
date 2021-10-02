from .functional import Functional, GradedFunctional
from .vect import Vect
from .sparse import eye

import torch

class Linear (Functional, Vect): 
    
    def __init__(self, domains, mat, name="Mat"):
        self.data = mat
        tgt = domains[-1]
        def matmul(x):
            return tgt.field(mat @ x.data)
        super().__init__(domains, matmul, name)

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
    
    def __init__(self, domains, mats, degree=0, name="Mat"):
        arr = [(domains[0][i], domains[-1][i + degree])\
                for i in range(len(mats))] 
        fs  = [Linear(arr[i], mi, name) \
                if not isinstance(mi, Linear) else mi\
                for i, mi in enumerate(mats)]
        super().__init__(domains, fs, degree, name)

    #--- Show ---

    def __repr__(self):
        return f"{self.degree} Linear {self}"
