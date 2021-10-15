import torch
import time

from topos.exceptions import VectError
    
class Vect: 
    """ A base container class for torch 1D-tensors """
    
    @classmethod
    def cast2(cls, u, v):
        """ 
        Return a triple (u', v', c) aimed at a target constructor c. 

        The pair (u', v') should represent composable
        numerical data that can be passed as c(u' `op` v'). 
        """
        if isinstance(v, (torch.Tensor, int, float)):
            return u.data, v, u.same
        if isinstance(v, u.__class__):
            if not isinstance(u.domain, tuple):
                if u.domain.size == v.domain.size:
                    return (u.data, v.data, u.same)
            else:
                same = not sum(not du.size == dv.size\
                            for du, dv in zip(u.domain, v.domain))
                if same:
                    return (u.data, v.data, u.same)
        raise VectError(
            "Could not cast to composable numerical data",
            f"invalid type pair {type(u), type(v)}")

    #--- Scalar product ---

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            return (self.data * other.data).sum()
        elif isinstance(other, Linear):
            return other.__class__(self.data @ other.data)

    #--- Arithmetic Operations ---

    def __add__(self, other): 
        a, b, c = self.cast2(self, other)
        return c(a + b)

    def __neg__(self):
        return self.same(- self.data)

    def __sub__(self, other): 
        a, b, c = self.cast2(self, other)
        return c(a - b)

    def __mul__(self, other): 
        a, b, c = self.cast2(self, other)
        return c(a * b)

    def __truediv__(self, other): 
        a, b, c = self.cast2(self, other)
        return c(a / b)

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

    def __str__(self):
        return self.name if name in self.__dir__()\
               else "v"
