import torch

matmul = torch.sparse.mm 

def irange (n): 
    return torch.arange(n, dtype=torch.long)

def eye(n): 
    return torch.sparse_coo_tensor(
        torch.stack([irange(n), irange(n)]),
        torch.ones([n]),
        [n, n]
    )

def sparse(shape, indices, values=1., t=True):
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


#--- Nerve --- 
"""
K = Set(cells)
chains = [[a.idx, b.idx] for a, b in K * K if a > b]
chains = chains if len(chains) else [[], []]

one = sparse.eye(len(K))

zeta_1 = torch.sparse_coo_tensor(
    torch.tensor(chains).t(),
    torch.ones([len(chains)]),
    (len(K), len(K))
)
zeta = one + zeta_1
self.zeta = zeta
"""

