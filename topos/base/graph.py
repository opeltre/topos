from topos.core import sparse, Shape
from topos.io   import alignString, parseTensor

import torch
from torch import stack, cat, arange

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

class Graph :
    """
    Hypergraphs.

    Given a set of vertices I, a hypergraph is a collection of subsets of I. 
    These subsets are called hyperdeges or more simply cells, regions, faces... 

    The degree or dimension of a face containing n vertices is n + 1.
    A graph is a hypergraph having only faces of dimension <= 1,
    though more precisely it is a simplicial complex of dimension 1.
    """
    
    def __init__(self, *grades, sort=True):
        """ 
        Construct hypergraph from lists of hyperedges by degrees.

        Example:
        -------
            G = Graph([[0], [1], [2]],
                      [[0, 1], [1, 2]])
        """
        
        G = ([parseTensor(js).sort(-1).values for js in grades] 
                if sort else [parseTensor(js) for js in grades])
        if len(G) and G[0].dim() == 1:
            G[0] = G[0].unsqueeze(1)
        
        #--- Adjacency and index tensors ---
        
        A, I, sizes = [], [], [] 
        i, Nvtx  = 0, 1 + max(Gi.max() for Gi in G)

        for k, Gk in enumerate(G):
            
            #--- sort and remove duplicates
            shape = [Nvtx] * (k + 1)
            Ak   = sparse.tensor(shape, Gk).coalesce()
            G[k] = Ak.indices().T
            Nk   = G[k].shape[0]
            A   += [sparse.tensor(shape, Ak.indices(), t=0).coalesce()]
            
            #--- index regions
            idx  = torch.arange(i, i + Nk)
            I   += [sparse.tensor(shape, G[k], idx).coalesce()]
            i   += Nk
            sizes += [Nk]

        #--- Attributes ---
        self.adj   = A
        self.idx   = [sparse.reshape([-1], Ik) for Ik in I]
        self.sizes = sizes
        self.Ntot  = i
        self.Nvtx  = A[0].shape[0]
        self.grades     = G
        self.vertices   = G[0].squeeze(1)
        self.dim        = len(G) - 1
       
    def adjacency(self, k):
        """ Symmetric adjacency tensor in degree k. """
        #--- symmetrize adjacency 
        shape = self.adj[k].size()
        Ak = sparse.tensor(shape, [])
        Sk = SymGroup(k + 1)
        for sigma in Sk:
            Ak += sparse.tensor(shape, self[k].index_select(1, sigma))
        return Ak

    def index (self, js):
        """ Index i of hyperedge [j0, ..., jn]. """
        js = parseTensor(js)
        n  = js.shape[-1]
        I = self.idx[n - 1]
        E = Shape(*([self.Nvtx] * n))
        i = E.index(js)
        return (I.index_select(0, i).to_dense() if i.dim() 
                else I.select(0, i))

    def coords(self, i, d=None):
        """ Hyperedge [j0, ..., jn] at index i. """
        i, begin = parseTensor(i), 0
        if not isinstance(d, type(None)):
            return self[d][i]
        for Gn in self.grades:
            i0 = i[0] if i.dim() == 1 else i
            if i0 - begin < Gn.shape[0]:
                return Gn[i - begin] 
            begin += Gn.shape[0]
    
    def nerve (self, d=-1):
        """ Categorical nerve of the hypergraph. """
        Ntot = self.Ntot 
        N = [torch.ones([Ntot]).to_sparse(),
             self.arrows()]
        arr = [N[1][i].coalesce().indices() for i in range(Ntot)]
        deg = 2
        if d == 1: return N
        while deg != d: 
            ijk = [cat([ij, k]) for ij in N[-1].indices().T \
                                for k in arr[ij[-1]].T]
            if not len(ijk): break
            Nd = sparse.matrix([Ntot] * (deg + 1), stack(ijk))
            N += [Nd.coalesce()]
            deg += 1
        return Nerve(*(Nd.indices().T for Nd in N), sort=False)

    def arrows (self): 
        """ 1-Chains of the hypergraph. """
        Ntot = self.Ntot
        N1   = sparse.matrix([Ntot, Ntot], [])
       
        A    = [sparse.reshape([-1], Ak) for Ak in self.adj]
        E    = [Shape(*Ak.size()) for Ak in self.adj]
        I    = self.idx

        for n, Gn in enumerate(self.grades):
            Nn = Gn.shape[0]

            # row indices
            i_ = I[n].index_select(0, E[n].index(Gn)).to_dense()
            # loop over subfaces
            F  = simplices(Gn)
            for k, Fk in enumerate(F[:-1]):
                # valid column indices
                nz = (A[k].index_select(0, E[k].index(Fk))
                          .to_dense()
                          .view([-1, Nn])
                          .nonzero())
                j_ = (I[k].index_select(0, E[k].index(Fk))
                          .to_dense()
                          .view([-1, Nn]))
                # ordered pairs
                ij = torch.tensor([[i_[y], j_[x, y]] for x, y in nz])
                N1 += sparse.matrix([Ntot, Ntot], ij)
        
        chains = N1.coalesce().indices()
        return sparse.matrix([Ntot, Ntot], chains, t=0).coalesce()

    def __getitem__(self, d):
        """ Degree d hyperedges [j0,...,jd] of the hypergraph. """ 
        return self.grades[d]
    
    def __repr__(self):
        return f"{self.dim} Graph {self}"
    
    def __str__(self):
        return self.show()

    def show (self, json=False):
        s = '{\n\n'
        degree = lambda d: (f'  {d}: ' if not json else f'  "{d}": ')
        s += alignString(degree(0), self[0]) + ',\n\n'
        for d in range(1, self.dim + 1):
            s += alignString(degree(d), self[d]) \
                + (',\n\n' if d < self.dim else '\n\n')
        return s + '}'


class Complex (Graph):
    """
    Simplicial complexes. 

    A simplicial complex is a hypergraph containing all the subsets of its cells. 
    The i-th face map consists of removing the i-th vertex from a degree-n cell,
    for 0 <= i <= n.
    Alternated sums of face maps induce a dual pair of differential and codifferential 
    (or boundary) operators, `d` and `delta = d*`. 

    References:
    -----------
    - Peltre, 2020:
        Message-Passing Algorithms and Homology, Chapter II
        https://arxiv.org/abs/2009.11631
    """

    def diff(self, d):
        """ Differential d: K[d] -> K[d + 1]. """
        src, tgt = self[d], self[d + 1]
        N, P = tgt.shape[0], src.shape[0]
        i = self.index(tgt) - self.index(tgt[0])
        out = sparse.matrix([N, P], [])
        for k in range(d + 2):
            j0  = self.index(src[0])
            j   = self.index(face(k, self[d + 1])) - j0
            val = (-1.) ** k
            out += sparse.matrix([N, P], stack([i, j]), val, t=0)
        return out
    
    @staticmethod
    def face(i, f):
        """ Face map acting on (batched) cells [j0, ..., jn]. """
        n = f.shape[-1]
        return f.index_select(f.dim() - 1, cat([arange(i), arange(i+1, n)]))
    
    @classmethod
    def simplicial(cls, faces):
        """ Simplicial closure. 
            
            Returns the complex containing every subface of the input faces.
        """
        faces = parseTensor(faces)
        K = [[(0, faces)]]
        def nextf (nfaces):
            faces = []
            for i, f in nfaces:
                faces += [(j, cls.face(j, f)) for j in range(i, f.shape[-1])]
            return faces
        for codim in range(faces.shape[-1] - 1):
            K += [nextf(K[-1])]
        K_ = [cat([f for i, f in Kn[::-1]]) for Kn in K[::-1]]
        return cls(*K_)
            
    def __repr__(self):
        return f'{self.dim} Complex {self}'


class Nerve (Complex):
    """
    Categorical nerves. 

    The nerve of a hypergraph is a simplicial complex. 
    The partial order structure of the hypergraph also
    induces a pair of combinatorial automorphisms, 
    extending the Zeta transform and Möbius inversion formula
    to the whole complex. 

    References:
    -----------
    - Rota, 1964: 
        On the Foundations of Combinatorial Theory - I. Theory of Möbius Transforms
    - Peltre, 2020: 
        Message-Passing Algorithms and Homology, Chapter III, 
        https://arxiv.org/abs/2009.11631
    """
    
    
    def zeta (self, d):
        """ Degree-d zeta transform. 

            See Nerve.zetas(d) to keep graded components until d.  
        """
        return self.zetas(d)[d]

    def zetas (self, d):
        """ List of zeta transforms in degrees <= d. """ 

        zt, N = [], [Nd.shape[0] for Nd in self.grades]

        # strict inclusions
        A  = self.adj[1]
        # weak inclusions
        zt0 = ( sparse.matrix([N[0], N[0]], A.indices(), t=False) 
              + sparse.eye(N[0])).coalesce()
        
        zt += [zt0]
        
        def next_diagrams (chain_i, chain_j):
            """ Extend diagrams to the right:

                    i0 > i1 > ... > id
                    >=   >=         >=
                    j0 > j1 > ... > jd
            """
            order = sparse.reshape([-1], zt0).coalesce()
            # Extend chain_i
            below_i = sparse.index_select(A, chain_i[-1])
            next_i  = below_i.indices()[1]
            deg_i   = sparse.sum_dense(below_i, [1]).long()
            chain_j = chain_j.repeat_interleave(deg_i, 1)
            chain_i = chain_i.repeat_interleave(deg_i, 1)
            chain_i = torch.cat([chain_i, next_i[None,:]])
            # Extend chain_j
            below_j = sparse.index_select(A, chain_j[-1])
            next_j  = below_j.indices()[1]
            deg_j   = sparse.sum_dense(below_j, [1]).long()
            chain_i = chain_i.repeat_interleave(deg_j, 1)
            chain_j = chain_j.repeat_interleave(deg_j, 1)
            chain_j = torch.cat([chain_j, next_j[None,:]])
            # Check that i[d] >= j[d] and i[d] !>= j[d-1]
            idx1  = N[0] * chain_i[-1] + chain_j[-1]
            idx2  = N[0] * chain_i[-1] + chain_j[-2]
            mask1 =   sparse.index_mask(order, idx1)
            mask2 = ~ sparse.index_mask(order, idx2)
            nz = (mask1 * mask2).nonzero().flatten()
            # Return diagrams
            return chain_i[:, nz], chain_j[:, nz]

        acc = [zt0.indices().unsqueeze(1)]

        for k in range(d):
            acc += [torch.stack(next_diagrams(*acc[-1]))]
        
        for chains, n, begin in zip(acc[1:], N[1:d+1], self.sizes[:d]):
            ij = torch.stack([self.index(chains[0].T) - begin, 
                              self.index(chains[1].T) - begin])
            zt += [sparse.tensor([n, n], ij, t=False)]

        return zt

    def __repr__(self):
        return f'{self.dim} Nerve {self}'
