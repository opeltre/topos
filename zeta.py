from simplex import Simplex
from hypergraph import Hypergraph
from tensor import Tensor, Product

K = Hypergraph(('i:j:k', 'i:k:l', 'j:k:l'))
K = K.closure()

""" Zeta Transform and Möbius Inversion """

# chains = {(a, b) | a > b in K * K} 

chains = (p for p in K * K if p[0] > p[1])

# zeta_ab = 1 if a >= b
#           0 otherwise

z = Tensor({
    (a, b): 1 for (a, b) in chains
}).curry()

I = Tensor({
    (a, a): 1 for a in K
}).curry()

zeta = I + z

# 1 / (1 + h) ~= sum_k (-1)**k h**k  for h << 1

def invert (a):
    h = (a - I).trim()
    hn, pows = I, []
    while not Tensor.iszero(hn):
        pows += [hn]
        hn = hn @ h
    return sum((-1)**k * hk for hk, k in Product(pows))

# mu @ zeta = I 

mu = invert(zeta)

""" Boundary and Coboundary """
