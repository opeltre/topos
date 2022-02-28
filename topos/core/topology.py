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

def simplices (faces):
    """ 
    Simplicial closure of a homogeneous collection of faces.
    
    If input is 1-dimensional or size d, returns all the k-faces 
    as a list of tensors of shape (nk, k).

    Given input (N, d), the output is a list of tensors 
    of shape (N, nk, k) for 0 < k <= d. 
    """
    faces   = readTensor(faces)
    batched = faces.dim() == 2
    n = faces.shape[0]
    K = [[(0, faces)]]
    def nextf (nfaces):
        """ join [[(j, dj(f)) | j >= i] | (i, dj) <- faces] """
        faces = []
        for i, f in nfaces:
            faces += [(j, face(j, f)) for j in range(i, f.shape[-1])]
        return faces
    for codim in range(faces.shape[-1] - 1):
        K += [nextf(K[-1])]
    F = []
    for d, Kd in enumerate(K[::-1]):
        Fd = torch.stack([fd for _, fd in Kd[::-1]])
        F += [Fd.transpose(0, 1) if batched 
                                 else Fd.view([-1, Fd.shape[-1]])]
    return F

