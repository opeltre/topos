from topos import Hypergraph, Chain, Cell, Shape
from .field import Field

import torch

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
        return Field(self, degree, torch.zeros([self.size[degree]]))

    def ones(self, degree=0):
        return Field(self, degree, torch.ones([self.size[degree]]))

    def __getitem__(self, chain): 
        return self.cells[Chain.read(chain)]

    def index(self, a, *js): 
        cell = self[a]
        return cell.begin + cell.shape.index(*js)

    def __repr__(self): 
        return f"System {self.cells}"


