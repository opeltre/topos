import torch
import sparse
from topos import Hypergraph, Set
from topos.hashable import Hashable

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
        for d in range(1, self.dim):
            i = (i * self.n[d]) + js[d]
        return i
    
    def __str__(self): 
        return "(" + ",".join([str(ni) for ni in self.n]) + ")"

    def __repr__(self): 
        return f"Shape {self}"

class Cell (Hashable): 

    def __init__(self, key, i, shape, start=0):
        self.key    = key 
        self.idx    = i
        self.start  = start
        self.end    = start + shape.size
        self.shape  = shape
        self.size   = shape.size

    def to (start):
        self.start = start
        self.end   = start + self.size

    def __gt__(self, other): 
        return self.key > other.key

    def __ge__(self, other): 
        return self.key >= other.key

    def __str__(self): 
        return str(self.key)


class System : 
    
    def __init__(self, K, shape=2, close=True, sort=True, degree=1):
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() if close else K
        E = lambda i: shape if type(shape) == int else shape[i]

        #--- Shapes of local tensors ---
        self.shape = { 
            a: Shape(*(E(i) for i in a)) for a in K
        }

        #--- Pointers to start of local data ---
        start, cells, cell_map = 0, [], {}
        for i, k in enumerate(K):
            c_k = Cell(k, i, self.shape[k], start = start)
            cells       += [c_k]
            cell_map[k]  = c_k
            start       += c_k.size

        self.size = start

        K = Set(cells)
        self.cells = K

        #--- Nerve --- 

        self.N = [K]
        chains = [[a.idx, b.idx] for a, b in K * K if a > b]
        chains = chains if len(chains) else [[], []]

        one = sparse.eye(len(K))

        zeta_1 = torch.sparse_coo_tensor(
            torch.tensor(chains).t(),
            torch.ones([len(chains)]),
            (len(K), len(K))
        )
        zeta = one + zeta_1
        self.zeta = zeta

    def index(self, a, *js): 
        return a.start + a.shape.index(*js)


class Field :

    def __init__(self, system, data=None):
        self.system = system
        if isinstance(data, torch.Tensor):
            self.data = data
        self.data = torch.zeros([system.size])

    def get(self, a):
        shape   = self.system.shape[a] 
        start   = self.system.start[a]
        end     = self.system.shape[a].size + start
        return self.data[start:end].view(shape)
    
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
