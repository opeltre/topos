from .system import System
import torch


def face(K, degree, j): 
    def dj (cell): 
        return K[degree - 1][cell.key.d(j)]
    pairs = [[dj(a), a] for a in K[degree]]
    fmap_is = eye_is if j < degree else functor_is
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


#------ Functorial Maps ------

def eye_is(cb, ca):
    """
    Identity indices from cell ca to cell cb (identical shapes).
    """
    return [[cb.begin + i, ca.begin + i] for i in range(cb.size)]

def functor_is(cb, ca):
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
    """ Position of elements of b in the inclusion b <= a. """
    a, b = a.list(), b.list()
    pos, j = [], 0
    for i, ai in enumerate(a):
        if j == len(b):
            break
        if ai == b[j]:
            pos += [i]
            j += 1
    return pos
