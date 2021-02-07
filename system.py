import torch 

from set import Set
from hypergraph import Hypergraph

from tensor import Tensor
from functional import Functional


class System (Hypergraph):

    def __init__(self, K, shape=2):

        if not isinstance(K, Hypergraph):
            K = Hypergraph(K)
        if not isinstance(K, System):
            K = K.closure()
        super().__init__((a for a in K))

        # E(i) = number of microstates for atom i
        E = lambda i: shape if type(shape) == int else shape[i]
        self.shape = {a: [E(i) for i in a] for a in self}
        
        # Nerve 
        I = self.field({
            (a, a) : 1 for a in K
        }).curry()
        below = self.field({
            (a, b) : 1 for a, b in K * K if a > b
        }).curry()

        self.below = below
        self.above = below.t()
        self.zeta = I + self.below 
   
    def field (self, *args):
        return Tensor(*args)
    
    def cofaces(self, n): 
        return [self.coface(n, i) for i in range(n + 2)]

    def coface(self, n, i): 
        cofaces = {b: [] for b in self[n]}
        for a in self[n + 1]:
            b = a.face(i)
            cofaces[b] += [a]
        return cofaces
    
    def zeros(self, face): 
        a = Set(face[-1])
        return torch.zeros(self.shape[a])
        
    def gaussian(self, face):
        a = Set(face[-1])
        return torch.randn(self.shape[a])

    def project(self, a, b):
        a, b = Set(a), Set(b)
        dim = tuple((i for i, x in enumerate(a) if x not in b))
        return lambda q_a: torch.sum(q_a, dim=dim)

    def extend(self, a, b):
        a, b = Set(a), Set(b)
        pull = [slice(None) if i in b else None for i in a]
        return lambda t_b : t_b[pull]
    
    def __getitem__(self, n): 
        if type(n) == tuple:
            Kn = (self.nerve(ni) for ni in n)
            return [a for a in product(*Kn)]
        return self.nerve(n)
    
    def __repr__(self): 
        elems = [str(e) for e in self]
        s = ' '.join(elems)
        return f"System {s}"
