from set import Set, Setmap
from simplex import Simplex
from hypergraph import Hypergraph
from tensor import Tensor

K = Hypergraph(('i:j:k', 'i:k:l', 'j:k:l'))
K = K.closure()

gt = lambda p: p[0] > p[1]

d1 = lambda s: (s[0], *s[2:])

p0 = lambda x: x[0]
p1 = lambda x: x[1]

chains = (K * K).fibers(gt)[True] 

cones = chains.curry()

N = [chains] 
"""
while len(N[-1]): 
    Nk = N[k-1].fmap(lambda bs: (*bs, c) for c in cones)
    N += [Nk]
"""

intervals = (K * K * K).fibers(d1)

zeta = Setmap({
    (a, b): 1 for (a, b) in chains
})

zeta_t = Setmap({(b, a): y for y, (a, b) in zeta})
