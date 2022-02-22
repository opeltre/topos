import torch
from math import pi
from topos.base import Shape

def matmul (A, B):
    """ Sparse matmul """
    mm = torch.sparse.mm

    if not(torch.is_complex(A) or torch.is_complex(B)):
        return mm(A, B)

    A, iA = from_complex(A)
    B, iB = from_complex(B)
    real = (mm(A, B)  - mm(iA, iB)).coalesce()
    imag = (mm(A, iB) + mm(iA,  B)).coalesce()

    return matrix(
        [A.shape[0], B.shape[1]],
        torch.cat([real.indices().t(), imag.indices().t()]),
        torch.cat([real.values().cfloat(), 1j * imag.values().cfloat()]),
    ).coalesce()

def irange (n): 
    return torch.arange(n, dtype=torch.long)

def diag (n, values): 
    return torch.sparse_coo_tensor(
        torch.stack([irange(n), irange(n)]),
        values,
        [n, n])

def zero(n, m): 
    return matrix([n, m], [])

def eye(n): 
    return diag(n, torch.ones([n]))

def matrix(shape, indices, values=1., t=True):
    if not len(indices):  
        indices = torch.tensor([[] for ni in shape], dtype=torch.long)
        t = False
    elif not isinstance(indices, torch.Tensor) and len(indices):
        indices = torch.tensor(indices, dtype=torch.long)
    if t: 
       indices = indices.t()
    if not isinstance(values, torch.Tensor):
        values = values * torch.ones([len(indices[0])])
    return torch.sparse_coo_tensor(indices, values, size=shape)

def tensor(shape, indices, values=1., t=True):
    return matrix(shape, indices, values, t)


#--- Efficient access to slices 

def row_select(g:torch.Tensor, rows:torch.LongTensor) -> torch.LongTensor:
    """ Select rows from a sparse matrix.

        Equivalent to `g.index_select(0, rows)` but much faster. 
        Prefer `g.to_dense()[rows]` when the size of `g` is not too large.
    """

    if not g.is_coalesced(): return row_select(g.coalesce(), rows)
    indices = g.indices()
    values  = g.values()

    # degree of the nodes
    deg_g = (torch.zeros(g.shape[0], dtype=torch.int32, device=indices.device)
                  .scatter_(0, indices[0], 1, reduce="add"))
    
    # row start indices
    query     = torch.arange(g.shape[0], dtype=torch.long)
    row_begin = torch.bucketize(query, indices[0])
    
    # edge indices mapped within shape N x max(deg_g)
    idx = (torch.arange(deg_g.max(), dtype=torch.long)
                    .unsqueeze(0)
                    .repeat(g.shape[0], 1))
    mask = idx < deg_g[:,None]
    idx += row_begin[:,None]

    # return stacked slices
    shape = (rows.shape[0], *g.shape[1:])
    val   = values[idx[rows][mask[rows]]]
    row_idx = torch.arange(shape[0]).repeat_interleave(deg_g[rows])
    col_idx = indices[1:, idx[rows][mask[rows]]]
    ij = torch.cat((row_idx[None,:], col_idx))
    return tensor(shape, ij, val, t=0)


#--- Index operations

def filter_idx(zt0:torch.Tensor, search_idx:torch.LongTensor) -> torch.BoolTensor:
    idx = zt0.indices().flatten()
    # find the position of the search_idx in zt0's indices
    pos_idx = torch.bucketize(search_idx,idx)
    # check if it match
    mask = idx[pos_idx]==search_idx

    return mask


#--- Reshape ---

def complete_shape(ns, shape):
    if -1 in ns:
        idx, prod = 0, 1
        for i, ni in enumerate(ns):
            if ni != -1: prod *= ni
            else:        idx   = i
        ns[idx] = shape.size // prod
    return ns

def reshape(ns, t):
    src = Shape(*t.size())
    tgt = Shape(*complete_shape(ns, src))
    t = t.coalesce()
    ij = tgt.coords(src.index(t.indices().T))
    return tensor(ns, ij, t.values())

#--- Fourier ---

def floordiv (a, b):
    return torch.div(a, b, rounding_mode='floor')

def iFT (N):
    T1 = torch.arange(N)
    T2 = torch.arange(N**2)
    kx = T1[:,None] * T1[None,:]
    ij = torch.stack([floordiv(T2, N), T2 % N])
    Fij = torch.exp(2j * pi * kx / N).view([-1])
    return ij, Fij

def FT (N):
    ij, Fij = iFT(N)
    return ij.flip(0), Fij.conj() / N

def iFourier (shape):
    N  = shape.size
    Ns = shape.ns
    Ts = shape.coords(torch.arange(N))
    ij = Shape(N, N).coords(torch.arange(N ** 2)).T
    kx = (Ts[:,None,:] * Ts[None,:,:] / Ns[None,None,:]).sum([-1])
    Fij = torch.exp(2j * pi * kx).view([-1])
    return ij, Fij

def Fourier (shape):
    ij, Fij = iFourier(shape)
    return ij.flip(0), Fij.conj() / shape.size

#--- Complex data ---

def from_complex (mat):
    if not torch.is_complex(mat):
        return mat, matrix(mat.shape, [])
    mat = mat.coalesce()
    ij  = mat.indices()
    Mij = mat.values()
    return (matrix(mat.shape, ij, Mij.real, t=0), 
            matrix(mat.shape, ij, Mij.imag, t=0))
