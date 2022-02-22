from topos.core import sparse
from topos.base import Shape

import torch
from torch import stack, cat, arange

from time import time

def alignString (alinea='', s='', prefix='tensor(', suffix=')'):
    return alinea + (str(s).replace(prefix, ' ' * len(prefix))
                           .replace(suffix, '')
                           .replace('\n', '\n' + ' ' * len(alinea)))

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

def face(i, f):
    n = f.shape[-1]
    return f.index_select(f.dim() - 1, cat([arange(i), arange(i+1, n)]))

def simplices (faces):
    faces = astensor(faces)
    K = [[(0, faces)]]
    def nextf (nfaces):
        faces = []
        for i, f in nfaces:
            faces += [(j, face(j, f)) for j in range(i, f.shape[-1])]
        return faces
    for codim in range(faces.shape[-1] - 1):
        K += [nextf(K[-1])]
    K_ = [cat([f for i, f in Kn[::-1]]) for Kn in K[::-1]]
    return K_

def Simplicial (faces): 
    return Graph(*simplices(faces))
        
class Graph :
    
    def __init__(self, *grades, sort=True):
        """ 
        Construct hypergraph from lists of hyperedges by degrees.

        Example:
        -------
            G = Graph([[0], [1], [2]],
                      [[0, 1], [1, 2]])
        """
        
        G = ([astensor(js).sort(-1).values for js in grades] 
                if sort else [astensor(js) for js in grades])
        if len(G) and G[0].dim() == 1:
            G[0] = G[0].unsqueeze(1)
        
        #--- Adjacency and index tensors ---
        
        A, I = [], [] 
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

        #--- Attributes ---
        self.adj   = A
        self.idx   = [sparse.reshape([-1], Ik) for Ik in I]
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
        js = astensor(js)
        n  = js.shape[-1]
        I = self.idx[n - 1]
        E = Shape(*([self.Nvtx] * n))
        i = E.index(js)
        return (I.index_select(0, i).to_dense() if i.dim() 
                else I.select(0, i))

    def coords(self, i, d=None):
        """ Hyperedge [j0, ..., jn] at index i. """
        i, begin = astensor(i), 0
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
        return self.grades[i]
    
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
   
    def diff(self, d):
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

    def __repr__(self):
        return f'{self.dim} Complex {self}'
 

class Nerve (Complex):
    
    def cones (self, d):
        A = self.adj[1]
        V = self.vertices
        N = self[d].shape[0]
        if d == 0:
            return cat([stack([V, V]).T, A.indices().T])
    
    def zeta (self, d):
        N = [Nd.shape[0] for Nd in self.grades]
        A = self.adj[1]

        ij  = A.indices().T
        zt0 = (sparse.matrix([N[0], N[0]], ij) 
              + sparse.eye(N[0])).coalesce()
    
        if d == 0:
            return zt0
        
        # i0 -> i1 
        # j0 -> j1
        #--- Yield arrows sourcing from j0 <= i0 ---
        edge0 = A.indices()

        lower = sparse.index_select(zt0, edge0[0]).indices()
        edge1 = sparse.index_select(A, lower[1]).indices()

        #--- Reindex arrows accordingly ---
        lower = lower[:,edge1[0]]
        edge0 = edge0[:,lower[0]]
        i0, i1  = edge0[0], edge0[1]
        j0, j1  = lower[1], edge1[1]

        #--- Check that i1 >= j1 and not i1 >= j0 ---
        zt0_ = sparse.reshape([-1], zt0).coalesce()
        mask1 = sparse.index_mask(zt0_, N[0] * i1 + j1)
        mask2 = sparse.index_mask(zt0_, N[0] * i1 + j0)
        mask  = mask1 * (~ mask2)
        
        #--- Remaining quadruples ---
        nz = mask.nonzero().view([-1])
        i0i1 = edge0.index_select(1, nz)
        j0j1 = stack([j0, j1]).index_select(1, nz)
        
        return cat([i0i1, j0j1])

    def __repr__(self):
        return f'{self.dim} Nerve {self}'

def otimes (A, B):
    A = A.coalesce() if not A.is_coalesced() else A
    B = B.coalesce() if not B.is_coalesced() else B
    #--- indices ---
    i, j = A.indices().T, B.indices().T
    I = i.repeat(j.shape[0], 1, 1)
    J = j.repeat(i.shape[0], 1, 1).transpose(0, 1)
    t0 = time()
    IJ = cat([I, J], 2)
    t1 = time()
    print(f'cat {I.shape}: {t1 - t0}')
    IJ = IJ.view([-1, IJ.shape[-1]])
    #--- values ---
    a, b = A.values(), B.values()
    ab = a[:,None] * b[None,:]
    #--- shape ---
    shape = A.size() + B.size()
    return sparse.tensor(shape, IJ, ab.view([-1]))

def rtimes (A, B):
    A = A.coalesce() if not A.is_coalesced() else A
    B = B.coalesce() if not B.is_coalesced() else B
    #--- indices ---
    i, j = A.indices().T, B.indices().T
    I = i.repeat(j.shape[0], 1, 1)
    J = j.repeat(i.shape[0], 1, 1).transpose(0, 1)
    t0 = time()
    IJ = cat([I, J], 2)
    t1 = time()
    print(f'cat {I.shape}: {t1 - t0}')
    IJ = IJ.view([-1, IJ.shape[-1]])
    #--- values ---
    x, y = A.values(), B.values()
    xy   = x[:,None] * y[None,:]
    #--- shape ---
    shape = A.size() + B.size()
    return sparse.tensor(shape, IJ, xy.view([-1]))

if __name__ == '__main__':
    import unittest

    class TestNerve(unittest.TestCase):

        @staticmethod
        def zeta(nerve, d):
            N = [Nd.shape[0] for Nd in nerve.grades]
            A = nerve.adj[1]

            ij  = A.indices().T
            zt0 = (sparse.matrix([N[0], N[0]], ij) 
                + sparse.eye(N[0])).coalesce()
        
            if d == 0:
                return zt0
            

            zt0_ = [zti.coalesce().indices()[0] for zti in zt0]
            
            # i0 -> i1 
            # j0 -> j1
            #t0 = time()
            #--- Yield arrows sourcing from j0 <= i0 ---
            edge0 = A.indices()

            lower = zt0.index_select(0, edge0[0]).coalesce().indices()
            edge1 = A.index_select(0, lower[1]).coalesce().indices()
            
            #--- Reindex arrows accordingly ---
            lower = lower[:,edge1[0]]
            edge0 = edge0[:,lower[0]]
            i0, i1  = edge0[0], edge0[1]
            j0, j1  = lower[1], edge1[1]

            #t1 = time()
            #print(f'lower: {t1 - t0}')

            #--- Check that i1 >= j1 and not i1 >= j0 ---
            #t0=time()
            zt0_ = sparse.reshape([-1], zt0)

            mask1 = zt0_.index_select(0, N[0] * i1 + j1).to_dense()
            mask2 = zt0_.index_select(0, N[0] * i1 + j0).to_dense()
            
            #t2 = time()
            #print(f'filter: {t2 - t1}')
            
            nz = (mask1 * (1-mask2)).nonzero().view([-1])
            
            #--- Remaining quadruples ---
            i0i1 = edge0.index_select(1, nz)
            j0j1 = stack([j0, j1]).index_select(1, nz)
            
            #t3 = time()
            #print(f'nonzero: {t3 - t2}')
            return cat([i0i1, j0j1])

        def test_zeta(self):
            G = Graph([0, 1, 2, 3], 
                    [[0, 1], [1, 2], [0, 2], [1, 3]],
                    [[0, 1, 2]])
                    
            N5 = G.nerve().nerve().nerve().nerve().nerve()

            valid = TestNerve.zeta(N5, 1)
            res = N5.zeta(1)

            self.assertTrue((valid==res).all())


    unittest.main()
