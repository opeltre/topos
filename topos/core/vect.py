import torch
    
class Vect: 
    """ A base container class for torch 1D-tensors """

    def same (self, data=None):
        if isinstance(data, type(None)):
            data = self.data
        return Vect(self, data)

    #--- Arithmetic Operations ---

    def __add__(self, other): 
        if not isinstance(other, Vect):
            other = self.same(other)
        return self.same(self.data\
            + other.data)

    def __sub__(self, other): 
        if not isinstance(other, Vect):
            other = self.same(other)
        return self.same(self.data\
            - other.data)

    def __mul__(self, other): 
        if not isinstance(other, Vect):
            other = self.same(other)
        return self.same(self.data\
            * other.data)

    def __div__(self, other): 
        if not isinstance(other, Vect):
            other = self.same(other)
        return self.same(self.data\
            / other.data)

    def __radd__(self, other): 
        return self.__add__(other)

    def __rsub__(self, other): 
        return self.__sub__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __iadd__(self, other):
        if not isinstance(other, Vect):
            other = self.same(other)
        self.data += other.data 
        return self

    def __isub__(self, other):
        if not isinstance(other, Vect):
            other = self.same(other)
        self.data -= other.data
        return self

    def __imul__(self, other):
        if not isinstance(other, Vect):
            other = self.same(other)
        self.data *= other.data
        return self

    def __idiv__(self, other):
        if not isinstance(other, Vect):
            other = self.same(other)
        self.data /= other.data
        return self
