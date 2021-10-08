import torch
    
class Vect: 
    """ A base container class for torch 1D-tensors """
    
    @classmethod
    def cast2(cls, u, v):
        if not isinstance(v, (cls, cls.__class__)):
            return (u, u.same(v), u)
        A, B = u.domain, v.domain
        return  (u, u.same(v), u) if A.size >= B.size else\
                (v.same(u), v, v) 
    
    #--- Scalar product ---

    def __matmul__(self, other):
        return (self.data * other.data).sum()

    #--- Arithmetic Operations ---

    def __add__(self, other): 
        a, b, c = self.cast2(self, other)
        return c.same(a.data + b.data)

    def __neg__(self):
        return self.same(- self.data)

    def __sub__(self, other): 
        a, b, c = self.cast2(self, other)
        return c.same(a.data - b.data)

    def __mul__(self, other): 
        a, b, c = self.cast2(self, other)
        return c.same(a.data * b.data)

    def __truediv__(self, other): 
        a, b, c = self.cast2(self, other)
        return c.same(a.data / b.data)

    def __radd__(self, other): 
        return self.__add__(other)

    def __rsub__(self, other): 
        return self.__sub__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __iadd__(self, other):
        other = self.same(other)
        self.data += other.data 
        return self

    def __isub__(self, other):
        other = self.same(other)
        self.data -= other.data
        return self

    def __imul__(self, other):
        other = self.same(other)
        self.data *= other.data
        return self

    def __itruediv__(self, other):
        other = self.same(other)
        self.data /= other.data
        return self
