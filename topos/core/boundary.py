from topos.base import Set, Hypergraph
from topos import System

class Boundary : 

    @classmethod
    def closure(cls, K, vertices):
        regions  = K[0].fibers.keys()
        b        = vertices if isinstance(vertices, Set)\
                    else Set(vertices) 
        boundary = Hypergraph((r & b for r in regions))
        B        = boundary.nerve(K)
        return cls(K, K.restriction(B))

    def __init__(self, K, B):
        ts = [K[i].res(B[i]) for i in range(K.degree + 1)]
        self.trace  = GradedLinear([K, B], ts, 0, "Res B")
        self.flux   = self.trace @ K.delta[1] 
        self.embed  = self.trace.t()
