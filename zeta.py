from simplex import Simplex
from hypergraph import Hypergraph
from tensor import Tensor, Product

K = Hypergraph(('i:j:k', 'i:k:l', 'j:k:l'))
K = K.closure()

""" Zeta Transform and Möbius Inversion """

# chains = {(a, b) | a > b} <= K * K

gt = lambda p: p[0] > p[1]

chains = (K * K).fibers(gt)[True] 

# zeta_ab = 1 if a >= b
#           0 otherwise

z = Tensor({
    (a, b): 1 for (a, b) in chains
}).curry()

I = Tensor({
    (a, a): 1 for a in K
}).curry()

zeta = I + z

# mu = inverse of zeta
# 1 / (1 + z) ~= sum_k (-1)**k z**k  for z << 1

zs = []
zn = I 
while not Tensor.iszero(zn): 
    zs += [zn]
    zn = zn @ z

mu = sum((-1)**k * zk for zk, k in Product(zs))

# mu @ zeta = I 

""" Boundary and Coboundary """
