import torch

def irange (n): 
    return torch.arange(n, dtype=torch.long)

def eye(n): 
    return torch.sparse_coo_tensor(
        torch.stack([irange(n), irange(n)]),
        torch.ones([n]),
        [n, n]
    )

