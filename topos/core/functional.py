from .field import Field
from .graded import Graded
import torch

class Functional :

    def __init__(self, domains, f, name="\u033b"):
        """ Create functional between domains = [src, tgt]. """
        self.src = domains[0] 
        self.tgt = domains[-1]
        self.domain = (self.src, self.tgt)
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

    def __init__(self, Ks, fs, degree=0, name="\u033b", cls=None):
        """ 
        Create a graded functional between complexes Ks = [src, target].
        """
        src, tgt, d = Ks[0], Ks[-1], degree
        Func        = Functional if not cls else cls
        self.degree = d
        self.grades = [
            Func([src[i], tgt[i + d]], fi, name = name)\
            if not isinstance(fi, Func) else fi\
            for i, fi in enumerate(fs)]

        def call (field):
            d = field.degree
            print(f"call at {d}")
            return self[d](field)

        super().__init__([src, tgt], call, name)

    @classmethod
    def map(cls, domains, fs, degree=0, name="\u033b"):
        """
        Create a graded functional from an array of torch.Tensor functions.
        """
        d = self.degree
        src, tgt = domains[0], domains[-1]
        map_fs = [lambda u : tgt[i + d].field(fi(u.data))\
                  for i, fi in enumerate(fs)]
        return cls([src, tgt], map_fs, name)
    
    #--- Components ---

    def null(self, d):
        src, tgt = [self.src[d], self.tgt[d + self.degree]]
        return self.null([src, tgt])
        
    def __getitem__(self, d):
        return  self.grades[d]\
                if d >= 0 and d < len(self.grades)\
                else self.null(d)        

    def items(self): 
        return enumerate(self.grades)

    #--- Call and Composition --- 
    
    def __call__(self, field):
        d = field.degree
        if d != None:
            return self[d](field)\
                   if (d >= 0 and d < len(self.grades))\
                   else self.tgt[d + self.degree].zeros()
        src, tgt = self.src, self.tgt
        return tgt.field(torch.cat(
            [(fd @ src.p(d) @ field).data for d, fd in self.items()]))

    def compose (self, other):
        src, tgt = other.src, self.tgt
        deg = self.degree + other.degree
        circ = [
            self[i + other.degree] @ gi\
            for i, gi in enumerate(other) 
            if i + other.degree < len(self.grades)]
        name = f"{self} . {other}"
        return self.__class__([src, tgt], circ, deg, name) 

