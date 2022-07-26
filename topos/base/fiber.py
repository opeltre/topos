from .domain    import Domain
from topos.io   import readTensor

import fp
import torch

class Fiber (Domain):
    """
    Elementary domain.

    The `Fiber([n0, ..., nd])` instance spans keys (x0, ..., xd)
    such that 0 <= xi < ni for all 0 <= i <= d. 

    """

    def __init__(self, key=None, shape=None, degree=None):
        self.trivial = isinstance(shape, type(None))
        if self.trivial:
            size  = 1
            self.shape = []
            self.dim = 0
        else:
            shape = readTensor(shape)
            size  = shape.prod().long()
            self.shape = shape.tolist()
            self.dim = shape.shape[0]
            self.ns  = shape
            self.mod = torch.tensor([shape[i+1:].prod() for i in range(self.dim)])
        #--- Domain attributes
        self.degree = degree
        self.size   = size
        super().__init__(size, degree)
    
    def index(self, keys, output=None):
        """
        Row-major index. 
        """
        if not len(keys):
            return 0
        js  = readTensor(keys)
        out = (self.mod * js).sum([-1])
        if type(output) == type(None):
            return out
        ns = self.ns[None,:] if js.dim() == 2 else self.ns 
        mask = ((js < ns).long().prod(1)
             *  (0 <= js).long().prod(1))
        return out, mask

    def coords(self, idx, output=None):
        """
        Coordinates of an index. 
        """
        div = lambda a, b: torch.div(a, b, rounding_mode = 'floor')
        idx = readTensor(idx, dtype=torch.long)

        i = 0 + (idx if i.dim() == 0 else idx[None,:])
        out = (torch.zeros([self.dim], dtype=torch.long)
                    if i.dim() == 0 else [None] * self.dim)

        for j, mj in enumerte(self.mod):
            out[j] = div(i, mj)
            i      = i % mj

        if i.dim() == 0: return out
        if not len(out): return torch.tensor([[]])
        return torch.stack(out).t()
    
    def res(self, indices):
        """ 
        Restriction to a subset of keys 0 <= kj < dim. 
        """
        pass

    def slices(self):
        yield (0, self.size, self)

    def slice(self, key=None):
        return (0, self.size, self)

    def __getitem__(self, key=None):
        return self

    def __iter__(self):
        yield self

    def items():
        yield self.key, self
