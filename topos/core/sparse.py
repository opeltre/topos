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

def index_select(g:torch.Tensor, idx:torch.LongTensor, dim=0) -> torch.Tensor:
    """ Select slices from a sparse matrix.

        Equivalent to `g.index_select(dim, idx)` but much faster. 
    """

    if not g.is_coalesced(): return index_select(g.coalesce(), idx, dim)

    indices = g.indices()
    values  = g.values()
    if dim != 0:
        sorting = indices[dim].sort().indices
        indices = indices[:,sorting]
        values  = values[sorting]

    # degree of the nodes
    deg_g = (torch.zeros(g.shape[dim], dtype=torch.int32, device=indices.device)
                  .scatter_(0, indices[dim], 1, reduce="add"))
    
    # slice start indices
    query     = torch.arange(g.shape[dim], dtype=torch.long)
    slice_begin = torch.bucketize(query, indices[dim])
    
    # edge indices mapped within shape N x max(deg_g)
    edges = (torch.arange(deg_g.max(), dtype=torch.long)
                .unsqueeze(0)
                .repeat(g.shape[dim], 1))
    mask = edges < deg_g[:,None]
    edges += slice_begin[:,None]

    # return stacked slices
    shape = [*g.shape[:dim], idx.shape[0], *g.shape[dim + 1:]]
    val   = values[edges[idx][mask[idx]]]
    ij    = indices[:, edges[idx][mask[idx]]]
    ij[dim] = torch.arange(shape[dim]).repeat_interleave(deg_g[idx])
    return torch.sparse_coo_tensor(ij, val, size=shape)

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
