import test
import torch

from topos.base import Graph, Nerve, FreeFunctor

G = Graph([[0],
           [[0, 1],[0, 2], [0, 3]],
           [[0, 1, 2], [0, 2, 3], [0, 1, 3]]])

N = Nerve.classify(G.quiver())

class TestNerve (test.TestCase):

    def test_zeta(self):
        zeta = N.zeta(0).data
        # zt0(x)[a] = sum [x[b] | b <= a]
        one = torch.ones([7])
        result = zeta @ one
        expect = torch.tensor([1, 2, 2, 2, 4, 4, 4])
        self.assertClose(expect, result)
        
    def test_zetas(self):
        zt = [lin.data for lin in N.zetas()]
        self.assertTrue(len(N) == len(zt))
        # zt1(y)[a0 > a1] =  sum[y[b0 > b1 | b0 <= a0, b1 <= a1, b0 !<= a1]
        one1 = torch.ones([3 + 3 + 6])
        result = zt[1] @ one1
        expect = torch.tensor(3*[1] + 3*[3] + 6*[3])
        self.assertClose(expect, result)
        # zt2 = id2 as 2 = len(N) is the maximal degree.
        id2 = torch.eye(6)
        self.assertClose(id2, zt[2].to_dense())

class TestNerveFunctor(test.TestCase):

    def test_classify(self):
        GF = Graph(G, FreeFunctor(2))
        NF = GF.nerve()
        self.assertEqual(NF.sizes[0], 2 + 4*3 + 8*3)
        self.assertEqual(NF.sizes[1], 2*6 + 4*6)
        self.assertEqual(NF.sizes[2], 2*6)
        # fmap . last
        Fij = NF.fmap(torch.tensor([[4, 1], [4, 0]]))
        Fij = (Fij.T - Fij.T[0]).T
        Fi = torch.arange(4)
        Fj = torch.arange(2).repeat_interleave(2)
        self.assertClose(Fij, torch.stack([Fi, Fj]))
        Fii = NF.fmap([[4, 2], [2]])
        Fii = Fii - Fii[:,0][:,None]
        self.assertClose(Fii, torch.arange(Fii.shape[1]).repeat(2, 1))

    def test_zeta(self):
        """ Functor-valued zeta commutes with scalar extension """
        GF = Graph(G, FreeFunctor(3))
        NF = GF.nerve()
        N = NF.scalars()
        # compute zeta[d] for all degrees
        zts = NF.zetas()
        # N0
        x = N.ones(0)
        j0 = NF[0].to_scalars().t()
        X0 = NF.zeta(j0 @ x)
        X1 = j0 @ N.zeta(x)
        self.assertClose(X0.data, X1.data)
        # N1 
        y = N.ones(1)
        j1 = NF.from_scalars(1)
        Y1, Y2 = NF.zeta(1)(j1 @ y), j1 @ N.zeta(y)
        self.assertClose(Y1.data, Y2.data)
        # N2
        zt2 = NF.zeta(2).data.to_dense()
        self.assertClose(zt2, torch.eye(zt2.shape[0]))

    def test_mu(self):
        """ Functor valued Möbius inversion """
        GF = Graph(G, FreeFunctor(3))
        N = G.nerve()
        # mu0
        mu0, zt0 = N.mu(0), N.zeta(0)
        result = (mu0 @ zt0).data.to_dense()
        expect = torch.eye(N[0].size)
        self.assertClose(expect, result)     
        # mu1
        mu1, zt1 = N.mu(1), N.zeta(1)
        result = (mu1 @ zt1).data.to_dense()
        expect = torch.eye(N[1].size)
        self.assertClose(expect, result)     
