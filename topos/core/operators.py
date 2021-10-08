import torch
from topos.core.sparse import matrix, eye, matmul, diag
from topos.base import Chain


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


#------ Functorial Maps ------

def pull_is(cb, ca, f):
    """
    Pullback indices from fiber ca to fiber cb by f_is.
    """
    return [[cb.begin + i, ca.begin + f(i)] for i in range(cb.size)]

def eye_is(cb, ca):
    """
    Identity indices from fiber ca to fiber cb (identical shapes).
    """
    return [[cb.begin + i, ca.begin + i] for i in range(cb.size)]
    return pull_is(cb, ca, None)

def extend_is(cb, ca):
    """ 
    Cylindrical extension indices from fiber ca to fiber cb. 
    """
    if cb.size == ca.size: 
        return eye_is(cb, ca)
    pos  = positions(cb.key[-1], ca.key[-1])
    return pull_is(cb, ca, cb.shape.res(*pos))


#------ Pullbacks ---

def pullback(A, B, f=None, fmap=None):
    """
    Pullback matrix of a map f: A -> B between domain keys.
    """
    f    = f if callable(f)       else lambda x:x
    fmap = fmap if callable(fmap) else lambda ca: lambda x:x
    indices = [ij for ca in A\
                  for ij in pull_is(ca, B.get(f(ca.key)), fmap(ca))]
    return matrix([A.size, B.size], indices)

#------ Pullback of `last : K[n] -> K[0]` ---

def pull_last(K, degree):
    """ 
    Map fields on K[0] to K[d] evaluating last element of the chain. 
    """
    if degree == 0:
        return eye(K[0].size)
    def last(chain):
        return ca.key[-1]
    return pullback(K[degree], K[0], last)


#------ Normalisation ------

def local_masses(domain):
    """
    Local normalisation factors, as block diagonal matrix of 1s.
    """
    indices = [[i, j] for a in domain\
                      for i in range(a.begin, a.end)\
                      for j in range(a.begin, a.end)]
    return torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long).t(),
        values = torch.ones([len(indices)]),
        size = [domain.size, domain.size])

def from_scalar(domain):
    """
    Extension from scalar fields to tensor valued fields. 
    """
    indices = [[i, a.idx] for a in domain\
                          for i in range(a.begin, a.end)]
    shape = [domain.size, len(domain.fibers)]
    return matrix(shape, indices)

def to_scalar(domain):
    """
    Local normalisation factors, as scalar values.
    """
    return from_scalar(domain).t()

#--- Restriction / Embedding ---

def restrict(domain, subdomain): 
    """
    Restriction matrix. 
    """
    pairs = [[cb, domain.fibers[cb.key]] for cb in subdomain]
    indices = [[cb.begin + i, ca.begin + i]\
                for cb, ca in pairs\
                for i in range(cb.size)]
    return matrix([subdomain.size, domain.size], indices)

#--- Differentials --- 

def face(K, degree, j): 
    """ Face map from K[d] to K[d - 1]. """
    def dj (fiber): 
        return K[degree - 1][fiber.key.d(j)]
    pairs = [[dj(a), a] for a in K[degree]]
    fmap_is = eye_is if j < degree else extend_is
    indices = [ij for p in pairs for ij in fmap_is(*p)]
    matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long).t(),
        values=torch.ones([len(indices)]),
        size=[K[degree - 1].size, K[degree].size]
    )
    return matrix

def coface(K, degree, j):
    return face(K, degree + 1, j).t()

def codifferential(K, degree): 
    """ Codifferential from K[d] to K[d - 1]. """
    shape = K[degree - 1].size, K[degree].size
    delta = torch.sparse_coo_tensor([[], []], [], shape)
    for j in range(0, degree + 1):
        delta += (-1)**j * face(K, degree, j)
    return delta

def differential(K, degree):
    """ Differential from K[d] to K[d + 1]. """
    return codifferential(K, degree + 1).t()

def nabla(K, degree, p):
    """ Differential with conditional expectations. """
    #--- conditional expectations on last coface ---
    n, m = K[degree].size, K[degree + 1].size
    weight   = pull_last(K, degree) @ p
    coweight = pull_last(K, degree + 1) @ (1 / p)
    mm = matmul
    dn = coface(K, degree, degree)
    dn_expect = mm(mm(diag(m, coweight), dn), diag(n, weight))
    #--- first cofaces ---
    d = coface(K, degree, 0)
    for j in range(1, degree + 1):
        d += (-1) ** j * coface(K, degree, j)
    return d + (-1) ** (degree + 1) * dn_expect
        
#------ Combinatorics ------

def zeta(K, degree):
    """ Zeta transform: automorphism of K[d]. """
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
        cd = [[Chain.read(ca), Chain.read(cb)] for ca, cb in chains[d]]
        fibers = [[K[d][ca], K[d][cb]] for ca, cb in cd]
        indices = [ij for p in fibers for ij in extend_is(*p)]
        n = K[d].size
        z += [matrix((n, n), indices)]
    return z

def invert_nil(mat, order=10, tol=1e-10):
    """ Invert 1 + N as 1 - N + N**2 - N**3 + ... """
    one = eye(mat.shape[0])
    x = mat - one 
    out = one - x
    k, xk = 2, matmul(x, x)
    while float(xk.norm()) > tol and k <= order:
        out += (-1)**k * xk
        xk = matmul(x, xk) 
        k += 1
    return out
