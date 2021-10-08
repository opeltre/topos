from .hashable  import Hashable
from .set       import Set
from .fiber     import Fiber

#--- Simplicial Keys ---

class Seq (Hashable):

    @classmethod
    def read(cls, arg):
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, str):
            elems = arg.split("|")
            return cls(*(e for e in elems))
        return cls(*arg)

    def __init__(self, *js): 
        self.degree = len(js) - 1
        self.elems = js
    
    def d (self, i): 
        return self.__class__(*(self.elems[:i] + self.elems[i+1:]))

    def __getitem__(self, i): 
        return self.elems[i % (self.degree + 1)]

    def __iter__(self):
        return self.elems.__iter__()

    def __str__(self): 
        return "|".join([str(e) for e in self.elems])

    def __repr__(self):
        return f"{self}"

class Chain (Seq): 

    @classmethod
    def read(cls, arg):
        if isinstance(arg, cls):
            return arg
        if isinstance(arg, str):
            elems = arg.split(" > ")
            return cls(*(Set(e) for e in elems))
        return cls(*arg)

    def __str__(self): 
        return " > ".join([str(e) for e in self.elems])


#--- Simplicial Fibers ---

class Sequence (Fiber):

    @staticmethod
    def read(key):
        if isinstance(key, Seq):
            return key
        else:
            return Seq.read(key)


class Simplex (Fiber):

    @staticmethod
    def read(key):
        if isinstance(key, Chain):
            return key
        else:
            return Chain.read(key)