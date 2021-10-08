from .sheaf import Sheaf
from topos.base import Shape, Fiber, Sequence
from topos.core import Field

from itertools import product

#--- Unit Object ---

class Point (Sheaf): 
    """ Point Domain spanning the field of scalars. """

    def __init__(self, degree=0):
        super().__init__(['\u2022'], None, degree)


#--- Null Object ---

class Empty (Sheaf):
    """ Empty Domain spanning the null vector space {0}. """
    
    def __init__(self, degree=0):
        super().__init__([], None, degree)

    def field(self, data=None, degree=0):
        return super().field([0.], self.degree)


#--- Product ---

class Product (Sheaf):
    """ Cartesian product of domains. """

    def __init__(self, *sheaves): 
        self.grades  = list(sheaves)
        self.rank    = len(sheaves) - 1
        self.trivial = not sum((not f.trivial for f in sheaves))
        self.scalars = self if self.trivial else\
                       self.__class__(*(f.scalars for f in sheaves))
    
        #--- Product of fibers --- 
        def key(*fibers):
            return Sequence.read([f.key for f in fibers])

        def shape(*fibers):
            out = []
            for f in fibers:
                out += f.shape.n
            return out
        shape = {
            key(*fs): shape(*fs) for fs in product(*sheaves)
        }
        super().__init__(shape, ftype=Sequence)

    def __getitem__(self, d=0):
        return self if d == None else\
               self.grades[d]
    
    def projection(self, d):
        def res_d (key):
            return key[d]
        return res_d
        
    def get(self, key, *keys):
        if isinstance(key, Sequence):
            return self.fibers[key]
        return super().get("|".join([key, *keys]))


#--- Coproduct: disjoint Unions --- 

class Sum (Sheaf):
    """ Disjoint union of domains. """

    def __init__(self, *sheaves):
        self.grades  = list(sheaves)
        self.rank    = len(sheaves) - 1
        self.trivial = not sum((not f.trivial for f in sheaves))
        self.scalars = self if self.trivial else\
                       self.__class__(*(f.scalars for f in self.grades))

        #--- Join fibers ---
        shape = {
            Sequence.read((str(i), a)): fa.shape \
                                        for i, Fi in enumerate(sheaves) \
                                        for a, fa in Fi.fibers.items()} 
        keys = shape.keys()
        shape = shape if not self.trivial else None
        super().__init__(keys, shape, ftype=Sequence)
        
    def __getitem__(self, d=0):
        return self if d == None else\
               self.grades[d]
    
    def embedding(self, d):
        def emb_d (k):
            return  Sequence.read((str(d), k))
        return emb_d

    def p (self, d):
        return self.pull(self[d], self.embedding(d), name="j*")
    
    def j (self, d):
        return self.p(d).t()

    def field(self, data, d=None):
        return Field(self, data, d) if d == None\
                else self[d].field(data, d)

    def get(self, key, d=None):
        if d == None:
            return self.fibers[Sequence.read(key)]
        return self[d].get(key)
