from .sheaf import Sheaf
from topos.base import Shape, Fiber, Simplex
from topos.core import Field

#--- Unit Object ---

class Point (Sheaf): 
    """ Point Domain spanning the field of scalars. """

    def __init__(self, degree=0):
        super().__init__([''], None, degree)


#--- Null Object ---

class Empty (Point):
    """ Empty Domain spanning the null vector space {0}. """

    def field(self, data=None, degree=0):
        return super().field([0.], self.degree)


#--- Disjoint Unions --- 

class Coprod (Sheaf):
    """ Disjoint union of domains. """

    def __init__(self, *sheaves):
        self.grades = list(sheaves)
        self.rank   = len(self.grades) - 1
        self.trivial = not sum((not f.trivial for f in sheaves))
        self.scalars = self if self.trivial else\
                       self.__class__(*(f.scalars for f in self.grades))

        #--- Join fibers ---
        shape = {Simplex.read((i, a)): \
            fa.shape for i, Fi in enumerate(sheaves) \
                     for a, fa in Fi.fibers.items()}

        super().__init__(shape.keys(), shape, ftype=Simplex)
  
    def trivialise (self): 
        return Coprod(*[F.scalar for F in self.grades])

    def __getitem__(self, d=0):
        return self if d == None else\
               self.grades[d]

    def field(self, data, d=None):
        return Field(self, data, d) if d == None\
                else self[d].field(data, d)

    def get(self, key, d=None):
        if isinstance(key, Simplex) and d == None:
            return super().get(key)
        return self[d].get(key)

