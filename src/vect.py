import torch
    
class Vect: 
    """ A base container class for torch 1D-tensors """

    def __init__(self, size, data):
        self.size = size
        self.data = data

    #--- Arithmetic Operations ---

    def same (self, data=None):
        if not isinstance(data, torch.Tensor):
            data = self.data
        return Vect(self, self.size, data)

    def __add__(self, other): 
        return self.same(self.data\
            + (other.data if isinstance(other, Vect) else other))

    def __sub__(self, other): 
        return self.same(self.data\
            - (other.data if isinstance(other, Vect) else other))

    def __mul__(self, other): 
        return self.same(self.data\
            * (other.data if isinstance(other, Vect) else other))

    def __div__(self, other): 
        return self.same(self.data\
            / (other.data if isinstance(other, Vect) else other))

    def __radd__(self, other): 
        return self.__add__(other)

    def __rsub__(self, other): 
        return self.__sub__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __iadd__(self, other):
        self.data += other.data if isinstance(other, Vect) else other
        return self

    def __isub__(self, other):
        self.data -= other.data if isinstance(other, Vect) else other
        return self

    def __imul__(self, other):
        self.data *= other.data if isinstance(other, Vect) else other
        return self

    def __idiv__(self, other):
        self.data /= other.data if isinstance(other, Vect) else other
        return self
