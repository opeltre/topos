import torch
import sparse
from topos import Hypergraph, Set, Chain
from topos.hashable import Hashable
from itertools import product


class Shape :

    def __init__(self, *ns):
        self.dim = len(ns)
        self.n = list(ns)
        size = 1
        for ni in ns:
            size *= ni
        self.size = size

    def index(self, *js):
        i = js[0]
        for d in range(1, min(len(js), self.dim)):
            i = (i * self.n[d]) + js[d]
        return i
    
    def __str__(self): 
        return "(" + ",".join([str(ni) for ni in self.n]) + ")"

    def __repr__(self): 
        return f"Shape {self}"

class Cell (Hashable): 

    def __init__(self, key, i, shape, begin=0):
        self.key    = key 
        self.idx    = i
        self.begin  = begin
        self.end    = begin + shape.size
        self.shape  = shape
        self.size   = shape.size

    def to (begin):
        self.begin = begin
        self.end   = begin + self.size

    def __gt__(self, other): 
        return self.key > other.key

    def __ge__(self, other): 
        return self.key >= other.key

    def __str__(self): 
        return str(self.key)
    
    def __repr__(self): 
        return f"Cell {self} {self.begin}-{self.end}"

class System : 
    
    def __init__(self, K, shape=2, close=True, sort=True, degree=-1):

        #--- Nerve ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() if close else K
        N = K.nerve(degree)
        if sort: 
            for Nk in N:
                Nk.sort(key = lambda c : (-len(c[-1]), str(c)))

        #--- Shapes of local tensors ---
        E = lambda i: shape if type(shape) == int else shape[i]
        self.shape = { 
            a: Shape(*(E(i) for i in a)) for a in K
        }

        #--- Pointers to start of local data ---
        self.size = []
        self.cells = {}
        self.nerve = [[] for Nk in N]
        for k, Nk in enumerate(N):
            begin = 0
            for i, c in enumerate(Nk):
                cell = Cell(c, i, self.shape[c[-1]], begin = begin)
                self.nerve[k]    += [cell]
                self.cells[c]     = cell
                begin            += cell.size
            self.size += [begin]

    def zeros(self, degree=0):
        return Field(self, torch.zeros([self.size[degree]]))

    def ones(self, degree=0):
        return Field(self, torch.ones([self.size[degree]]))

    def __getitem__(self, chain): 
        return self.cells[Chain.read(chain)]

    def index(self, a, *js): 
        cell = self[a]
        return cell.begin + cell.shape.index(*js)

    def __repr__(self): 
        return f"System {self.cells}"


class Field :

    def __init__(self, system, data=None):
        self.system = system
        if isinstance(data, torch.Tensor):
            self.data = data
        self.data = torch.zeros([system.size])

    def get(self, a):
        shape   = self.system.shape[a] 
        begin   = self.system.begin[a]
        end     = self.system.shape[a].size + begin
        return self.data[begin:end].view(shape)
    
    #--- Arithmetic Operations ---

    def __add__(self, other): 
        return Field(self, self.data\
            + other.data if isinstance(other, Field) else other)

    def __sub__(self, other): 
        return Field(self, self.data\
            - other.data if isinstance(other, Field) else other)

    def __mul__(self, other): 
        return Field(self, self.data\
            * other.data if isinstance(other, Field) else other)

    def __div__(self, other): 
        return Field(self, self.data\
            / other.data if isinstance(other, Field) else other)

    def __radd__(self, other): 
        return self.__add__(other)

    def __rsub__(self, other): 
        return self.__sub__(other)

    def __rmul__(self, other): 
        return self.__mul__(other)

    def __iadd__(self, other):
        self.data += other.data if isinstance(other, Field) else other
        return self

    def __isub__(self, other):
        self.data -= other.data if isinstance(other, Field) else other
        return self

    def __imul__(self, other):
        self.data *= other.data if isinstance(other, field) else other
        return self

    def __idiv__(self, other):
        self.data /= other.data if isinstance(other, field) else other
        return self
