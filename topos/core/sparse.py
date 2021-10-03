import torch

matmul = torch.sparse.mm 

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
