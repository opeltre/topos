from .shape import Shape

import torch
from math import pi

def matmul (A, B):
    """ Sparse matmul, accepting complex input. """
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

#--- Constructors ---

def irange (n):
    return torch.arange(n, dtype=torch.long)

def diag (n, values):
    """ Diagonal matrix. """
    return torch.sparse_coo_tensor(
        torch.stack([irange(n), irange(n)]),
        values,
        [n, n])

def zero(n, m):
    """ Zero matrix. """
    return matrix([n, m], [])

def eye(n):
    """ Identity matrix. """
    return diag(n, torch.ones([n]))

def matrix(shape, indices, values=1., t=True):
    """ Alias of sparse.tensor. """
    if not len(indices):
        indices = torch.tensor([[] for ni in shape], dtype=torch.long)
        t = False
    elif not isinstance(indices, torch.Tensor) and len(indices):
        indices = torch.tensor(indices, dtype=torch.long)
    if t:
       indices = indices.T
    if not isinstance(values, torch.Tensor):
        values = values * torch.ones([len(indices[0])])
    return torch.sparse_coo_tensor(indices, values, size=shape)

def tensor(shape, indices, values=1., t=True):
    """ Sparse tensor constructor.

        The table of indices is assumed transposed by default (t=True). """
    return matrix(shape, indices, values, t)

#--- Efficient access to slices

def index_select(g:torch.Tensor, idx:torch.LongTensor, dim:int = 0) -> torch.Tensor:
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
    return torch.sparse_coo_tensor(ij, val, size=shape).coalesce()

#--- Efficient access to values

def select(x:torch.Tensor, idx:torch.LongTensor) -> torch.Tensor:
    """ Select pointwise values from a sparse tensor. """
    if x.dim() > 1:
        shape  = Shape(*[ni for ni in x.shape])
        x_flat = reshape([-1], x)
        return select(x_flat, shape.index(idx))
    if idx.dim() > 1:
        idx = idx.view([-1])
    if not x.is_coalesced():
        x = x.coalesce()
    supp = x.indices().flatten()
    val  = x.values()
    pos  = torch.bucketize(idx, supp)
    pos  = torch.min(pos, torch.tensor(supp.shape[0] - 1))
    mask = (supp[pos] == idx)
    return mask * val[pos]
      
#--- Filtering indices

def index_mask(x:torch.Tensor, idx:torch.LongTensor) -> torch.BoolTensor:
    """
    Mask indices not present in a sparse tensor.
      
    Returns a vector of size `idx.shape[0]`,
    assuming `idx.shape[1] == x.dim()`.
    """
    if not (x.dim() == 1 and idx.dim() == 1):
        x_flat = reshape([-1], x).coalesce()
        idx_flat = Shape(*x.shape).index(idx)
        return index_mask(x_flat, idx_flat)
    indices = x.indices().flatten()
    # Closest position of index in the indices of x.
    pos = torch.bucketize(idx, indices)
    pos = torch.min(pos, torch.tensor(indices.shape[0] - 1))
    # Check that indices match
    return idx == indices[pos]

#--- Sum slices

def sum_dense (x:torch.Tensor, dim:list) -> torch.Tensor:
    """ Sum values along given dimensions. """
    mask = torch.tensor([True] * x.dim())
    dims = torch.tensor(dim, dtype=torch.long)
    mask[dims] = False
    tgt = Shape(*[x.shape[d] for d, md in enumerate(mask) if md])
    val = x.values()
    idx = x.indices()
    return (torch.zeros([tgt.size])
                .scatter_(0, tgt.index(idx[mask].T), 1, reduce="add")
                .view(tgt.ns))

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
