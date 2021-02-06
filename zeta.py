from hypergraph import Hypergraph
from tensor import Tensor, Product, Matrix
from functional import Lambda, Functional

K = Hypergraph(('i:j:k', 'i:k:l', 'j:k:l'))
K = K.closure()

""" Zeta Transform and Möbius Inversion """

# chains = {(a, b) | a > b in K * K} 

chains = [p for p in K * K if p[0] > p[1]]

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

def invert (x, one=I):
    h = (x - one).trim()
    hn, pows = one, []
    while not Tensor.iszero(hn):
        pows += [hn]
        hn = hn @ h
    return sum((-1)**k * hk for hk, k in Product(pows))

# mu @ zeta = I 

mu = invert(zeta)

""" Higher degree combinatorics """

# (t >> s)_ab = t_a * s_b for a[-1] > b[0]

def cup(t, s):
    r = {}
    prod = lambda ti, sj: cup(ti, sj) if\
        isinstance(ti, Tensor) and isinstance(sj, Tensor) \
        else ti * sj
    s_ = s.fibers(lambda sb, b: b.project(0))
    for ta, a in t:
        i = a.project(-1)
        for _, j in z[i] if i in z else []:
            for sb, b in s_[j] if j in s_ else []:
                r[a|b] = prod(ta, sb)
    return t.__class__(r).trim()

# Zeta[n] : zeta-transform on K[n]

def cup_pows(t):
    pows = []
    tn = t
    while not Tensor.iszero(tn): 
        pows += [tn]
        tn = cup(tn, t)
    return Product(pows)

Zeta = cup_pows(zeta)
Id = cup_pows(I)
Mu = Zeta.map(lambda zn, n: invert(zn, Id[n]))

""" Nerve """ 

N0 = Tensor({(a, ): 1 for a in K})
N = cup_pows(N0)

""" Boundary and Coboundary """
# d = sum_i (-1)**i di 

def diff(n): 
    if not n < len(N) - 1:
        return Matrix()
    face = lambda i: Matrix({
        a : {a.forget(i) : 1} for _, a in N[n + 1]
    })
    d = sum((-1)**i * face(i) for i in range(n + 2))
    return d

def codiff(n): 
    if not n > 0:
        return Matrix()
    return diff(n - 1).t()

d = Product(diff(i) for Ni, i in N)
delta = Product(codiff(i) for Ni, i in N)

""" Laplacian """ 
# L = d d* + d* d 

D = sum(di for di, i in d)

# Laplacian = D @ D.t + D.t @ D

mul_ = lambda A, B: Matrix({ 
    i: {j: Ai & Bj for Bj, j in B} for Ai, i in A
})

D_t = D.t()
Laplacian = mul_(D, D) + mul_(D_t, D_t)

""" Graded Components """
# L[i] : A(N[i]) -> A(N[i])

NoN = Product((Ni | Ni).curry(range(i + 1)) for Ni, i in N)

L = Product(Laplacian * mi for mi, i in NoN)

""" Functionals and Operators """

Id = Functional({(a, a): 1 for a in K})

def extend(a, b):
    pull = [slice(None) if i in b else None for i in a]
    j_ab = lambda tb: tb[pull]
    j_ab.__name__ = f"Extend {a} < {b}"
    return j_ab

J = Id + Functional({
    (a, b): extend(a, b) for (a, b) in chains
}) 

def project(a, b):
    dim = tuple((i for i, x in enumerate(a) if x not in b))
    S_ab = lambda qa: torch.sum(qa, dim=dim)
    S_ab.__name__ = f"Sum {a} > {b}"
    return S_ab

Sigma = Id + Functional({
    (a, b): project(a, b) for (a, b) in chains
}) 

def Cofunctor (matrix): 
    F = matrix.map(
        lambda Ma, a: Ma.map(
        lambda Mab, b: Mab * J[a.p(-1)|b.p(-1)]
    ))
    return Functional(F)

def Functor (matrix): 
    F = matrix.map(
        lambda Mb, b: Mb.map(
        lambda Mba, a: Mba * Sigma[a.p(-1)|b.p(-1)]
    ))
    return Functional(F)


K.zeta = Zeta.fmap(Cofunctor)
K.mu = Mu.fmap(Cofunctor)
K.d = d.fmap(Functor)
K.delta = delta.fmap(Cofunctor)
