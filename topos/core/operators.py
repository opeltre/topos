import torch
from topos.core.sparse import sparse, eye, matmul

#--- Differentials --- 

def face(K, degree, j): 
    def dj (cell): 
        return K[degree - 1][cell.key.d(j)]
    pairs = [[dj(a), a] for a in K[degree]]
    fmap_is = eye_is if j < degree else extend_is
    indices = [ij for p in pairs for ij in fmap_is(*p)]
    matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long).t(),
        values=torch.ones([len(indices)]),
        size=[K[degree - 1].size, K[degree].size]
    )
    return matrix

def codifferential(K, degree): 
    shape = K[degree - 1].size, K[degree].size
    delta = torch.sparse_coo_tensor([[], []], [], shape)
    for j in range(0, degree + 1):
        delta += (-1)**j * face(K, degree, j)
    return delta

def differential(K, degree):
    return codifferential(K, degree + 1).t()


#------ Combinatorics ------

def zeta(K, degree):
    G = K.hypergraph
    chains = [[] for d in range(degree + 1)]
    chains[0] = [[[a], [b]] for a in G for b in G.below(a, strict=0)]
    for d in range(1, degree + 1): 
        chains[d] = [[[a0] + a1s, [b0] + b1s]   \
                    for a1s, b1s in chains[d-1] \
                    for a0 in G.above(a1s[0])   \
                    for b0 in G.above(b1s[0])   \
                    if (b0 <= a0 and not b0 <= a1s[0])]
    z = []
    for d in range(0, degree + 1):
        cells = [[K[d][ca], K[d][cb]] for ca, cb in chains[d]]
        indices = [ij for p in cells for ij in extend_is(*p)]
        n = K[d].size
        z += [sparse((n, n), indices)]
    return z

def invert_nil(mat, order=10, tol=1e-10):
    one = eye(mat.shape[0])
    x = mat - one 
    out = one - x
    k, xk = 2, matmul(x, x)
    while float(xk.norm()) > tol and k <= order:
        out += (-1)**k * xk
        xk = matmul(x, xk) 
        k += 1
    return out

#------ Normalisation ------

def local_masses(domain):
    indices = [[i, j] for a in domain\
                      for i in range(a.begin, a.end)\
                      for j in range(a.begin, a.end)]
    return torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long).t(),
        values = torch.ones([len(indices)]),
        size = [domain.size, domain.size])

#------ Functorial Maps ------

def eye_is(cb, ca):
    """
    Identity indices from cell ca to cell cb (identical shapes).
    """
    return [[cb.begin + i, ca.begin + i] for i in range(cb.size)]

def extend_is(cb, ca):
    """ 
    Cylindrical extension indices from cell ca to cell cb. 
    """
    pos = positions(cb.key[-1], ca.key[-1])
    def index_b(ia): 
        xs = cb.shape.coords(ia)
        ys = [xs[p] for p in pos]
        return ca.shape.index(*ys)
    return [[cb.begin + i, ca.begin + index_b(i)] for i in range(cb.size)]


#------ Canonical Projection ------

def positions(a, b): 
    """ Position of elements of b in the set inclusion b <= a. """
    a, b = a.list(), b.list()
    pos, j = [], 0
    for i, ai in enumerate(a):
        if j == len(b):
            break
        if ai == b[j]:
            pos += [i]
            j += 1
    return pos
