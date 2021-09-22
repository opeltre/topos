import torch

def irange (n): 
    return torch.arange(n, dtype=torch.long)

def eye(n): 
    return torch.sparse_coo_tensor(
        torch.stack([irange(n), irange(n)]),
        torch.ones([n]),
        [n, n]
    )

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

