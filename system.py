import torch 

from set import Set
from hypergraph import Hypergraph
from simplex import Simplex 

class System (Hypergraph):

    def __init__(self, hypergraph, shape=2, free="True"): 
        super().__init__(hypergraph)
        self.close()
        if type(shape) == int: 
            Ni = shape
            shape = {a: [Ni for i in a] for a in self}
        elif free == "True": 
            shape = {a: [shape[i] for i in a] for a in self}
        self.shape = shape 
    
    def zeros(self, face): 
        a = Set(face[-1])
        return torch.zeros(self.shape[a])
        
    def gaussian(self, face):
        a = Set(face[-1])
        return torch.randn(self.shape[a])

    def project(self, a, b):
        a, b = Set(a), Set(b)
        dim = (i for i in a if i not in b) 
        return lambda q_a: torch.sum(q_a, dim=dim)

    def extend(self, a, b):
        a, b = Set(a), Set(b)
        pull = [slice(None) if i in b else None for i in a]
        return lambda t_b : t_b[pull]

    def __getitem__(self, n): 
        return self.nerve(n)
