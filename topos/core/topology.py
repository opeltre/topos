import torch
from torch import cat, stack, arange

from topos.io import readTensor

def face(i, f):
    """ 
    Face map acting on (batched) cells [j0, ..., jn]. 

    Removes the i-th vertex of j, in the last dimension. 

        F_i(j) = [*j[0:i], *[j[i+1:]]
    """
    n = f.shape[-1]
    return f.index_select(f.dim() - 1, cat([arange(i), arange(i+1, n)]))

def simplices (faces, indices=False):
    """ 
    Simplicial closure of a homogeneous collection of faces.
    
    If input is 1-dimensional or size d, returns all the k-faces 
    as a list of tensors of shape (nk, k).

    Given input (N, d), the output is a list of tensors 
    of shape (N, nk, k) for 0 < k <= d.

    If `indices=True` then a list of tuples of size nk is also returned,
    representing the forgotten indices of the nk degree k subfaces. 
    """
    faces   = readTensor(faces)
    batched = faces.dim() == 2
    n = faces.shape[0]
    # last forgotten index and face
    K = [[faces]]
    # forgotten indices
    J = [[()]]

    def nextf (faces_1, idx_1):
        """ join [[(j, dj(f)) | j >= i] | (i, dj) <- faces] """
        faces = []
        idx   = []
        for f, js in zip(faces_1, idx_1):
            j1 = js[-1] if len(js) else -1
            i1 = j1 - len(js) + 1
            faces += [face(i, f) for i in range(i1, f.shape[-1])]
            idx   += [tuple([*js, i + len(js)]) for i in range(i1, f.shape[-1])]
        return faces, idx

    for codim in range(faces.shape[-1] - 1):
        K_1, J_1 = nextf(K[-1], J[-1])
        K += [K_1]
        J += [J_1]

    F = []
    for d, Kd in enumerate(K[::-1]):
        Fd = torch.stack(Kd[::-1])
        F += [Fd.transpose(0, 1) if batched 
                                 else Fd.view([-1, Fd.shape[-1]])]

    if not indices:
        return F
    Idx = []
    for d, Jd in enumerate(J[::-1]):
        Idx += [Jd[::-1]]
    return F, Idx

