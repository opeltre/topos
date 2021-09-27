from .field import Field
import torch


class Functional :

    @classmethod
    def map(cls, f, degree=0, name="\u033b"):
        def map_f(field):
            target = field.system[field.degree + degree]
            data = f(field.data)
            return target.field(data)
        return cls(map_f, degree, name)

    def __init__(self, f, degree=0, name="\u033b"):
        self.call = f if callable(f) else lambda x : x.same(f)
        self.degree = degree
        self.name = name

    def __call__(self, field):
        return self.call(field)

    def __matmul__(self, other): 
        if isinstance(other, Field):
            return self(other)
        if isinstance(other, Functional):
            degree = self.degree + other.degree
            def compose (data):
                return self.call(other.call(data))
            name = f"{self.name} . {other.name}"
            return self.__class__(compose, degree, name)

    def rename(self, name):
        self.name = name
        return self
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.degree} Functional {self}"
