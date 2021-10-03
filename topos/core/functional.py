from .field import Field
from .graded import Graded
import torch

class Functional :

    def __init__(self, domains, f, name="\u033b"):
        """ Create functional between domains = [src, tgt]. """
        self.src = domains[0] 
        self.tgt = domains[-1]
        self.call = f if callable(f) else lambda x : tgt.field(f)
        self.name = name

    @classmethod
    def map(cls, domains, f, name="\u033b"):
        """ Create functional from a torch.Tensor function. """
        target = domains[-1]
        def map_f(field):
            data = f(field.data)
            return target.field(data)
        return cls(domains, map_f, name)

    @classmethod 
    def unit (cls, domains, y, name="\u033b"):
        target = domains[-1]
        name = (str(y), name)[isinstance(y, (Field, torch.Tensor))]
        data = y.data if isinstance(y, Field) else y
        def ret_y(field):
            return target.field(data)
        return cls(domains, ret_y, name)
    
    @classmethod 
    def null (cls, domains):
        return cls.unit(domains, 0)

    #--- Call and Composition ---

    def __call__(self, field):
        """ Action on fields. """
        return self.call(field)

    def compose(self, other):
        """ Composition of functionals. """
        domains = [other.src, self.tgt]
        def circ (x):
            return self(other(x))
        name = f"{self.name} . {other.name}"
        return Functional(domains, circ, name)

    def __matmul__(self, other): 
        """ Composition of functionals / action on fields. """
        if isinstance(other, Field):
            return self(other)
        if isinstance(other, Functional):
            return self.compose(other)
    
    #--- Show ---

    def rename(self, name):
        self.name = name
        return self
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Functional {self}"


class GradedFunctional (Graded, Functional):

    def __init__(self, Ks, fs, degree=0, name="\u033b"):
        """ 
        Create a graded functional between complexes Ks = [src, target].
        """
        self.src = Ks[0]
        self.tgt = Ks[-1]
        self.degree = degree
        arr = [[self.src[i], self.tgt[i + degree]]\
               for i in range(len(fs))]
        self.grades = [
            Functional(arr[i], fi, name = f"{name}[{i}]")\
            if not isinstance(fi, Functional) else fi\
            for i, fi in enumerate(fs)]
        self.name = name

    @classmethod
    def map(cls, domains, f, name="\u033b"):
        """ Create functional from a torch.Tensor function. """
        target = domains[-1]
        def map_f(field):
            data = f(field.data)
            return target.field(data)
        return cls(domains, map_f, name)
    
    #--- Components ---

    def null(self, d):
        src, tgt = [self.src[d], self.tgt[d + self.degree]]
        return Functional.null([src, tgt])
        
    def __getitem__(self, d):
        return  self.grades[d]\
                if d >= 0 and d <= self.degree\
                else self.null(d)        

    #--- Call and Composition --- 
    
    def __call__(self, field):
        d = field.degree
        return self[d](field)\
               if (d > 0 and d <= self.degree)      \
               else self.tgt[d + self.degree].zeros()

    def compose (self, other):
        src, tgt = other.src, self.tgt
        deg = self.degree + other.degree
        circ = [
            self[i + other.degree] @ gi\
            for i, gi in enumerate(other) 
            if i + other.degree < len(self.grades)]
        name = f"{self} . {other}"
        return self.__class__([src, tgt], circ, deg, name) 
