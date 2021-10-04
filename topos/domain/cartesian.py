from .domain import Domain 
from topos.base import Shape, Cell

#--- Scalar Domain --- 

class Trivial (Domain) :
    """ Domain with point fibers. """

    def __init__(self, keys, degree=0):
        self.degree = degree
        self.shape = {k: Shape() for k in keys}
        self.cells, self.size = join_cells(keys, self.shape)

#--- Unit Object ---

class Point (Domain): 
    """ Point Domain spanning field of scalars R. """

    def __init__(self, degree=0):
        super().__init__({'()': Cell('()', 0, Shape())}, degree)


#--- Null Object ---

class Empty (Point):
    """ Empty Domain spanning the null vector space {0}. """

    def field(self, data=None, degree=0):
        return super().field([0.], self.degree)

#--- Disjoint Unions --- 
