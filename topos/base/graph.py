from topos.core import sparse

import torch
from torch import stack, cat

def SymGroup (n):
    if n == 1:
        return torch.tensor([[0]])
    if n == 2:
        return torch.tensor([[0, 1], [1, 0]])
    Sn_1 = SymGroup(n - 1)
    last = torch.tensor([n-1])
    return cat([stack([
        cat([s[:n-1-i], last, s[n-1-i:]]) for i in range(n) \
    ]) for s in Sn_1 ])

def astensor (js):
    return torch.tensor(js) if not isinstance(js, torch.Tensor) else js

class Graph :
    
    def __init__(self, *grades):
        
        G = [astensor(js).sort(-1).values for js in grades]
        if len(G) and G[0].dim() == 1:
            G[0] = G[0].unsqueeze(1)
        
        self.grades     = G
        self.vertices   = G[0].squeeze(1)
        self.dim        = len(G)
       
        #--- Index regions ---
        
        Nvtx  = 1 + max(Gi.max() for Gi in G)
        i, I = 0, []
        for i, Gi in enumerate(G):
            val  = torch.arange(i, i + Gi.shape[0])
            I   += [sparse.tensor([Nvtx] * (i+1), Gi, val).coalesce()]
            i   += Gi.shape[0]
        self.index = I
        self.Ntot  = i
        
        #--- Adjacency tensors ---

        A = []
        for i, Gi in enumerate(G):
            shape = [Nvtx] * (i+1)
            Si    = SymGroup(i+1)
            Ai    = sparse.tensor(shape, Gi)
            for sigma in Si[1:]:
                Ai += sparse.tensor(shape, Gi.index_select(1, sigma))
            A += [Ai.coalesce()]
        self.adj = A
        
    def __getitem__(self, i):
        return self.grades[i]
        
