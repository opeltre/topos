import torch 

def SymGroup (n):
    """ 
    Symmetric group of permutations.

    Returns a size n! x n tensor enumerating
    all possible permutations of {0, ..., n-1}. 
    """
    if n == 1:
        return torch.tensor([[0]])
    if n == 2:
        return torch.tensor([[0, 1], [1, 0]])
    Sn_1 = SymGroup(n - 1)
    last = torch.tensor([n-1])
    return torch.cat([torch.stack([
        torch.cat([s[:n-1-i], last, s[n-1-i:]]) for i in range(n) \
    ]) for s in Sn_1 ])

