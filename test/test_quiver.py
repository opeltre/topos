from topos import Quiver, FreeFunctor, Graph

import torch
import fp
import test

# objects : 0..3
Q0 = torch.arange(4)

# arrows : fk : i -> j  <=>  [i, j, k] 
Q1 = torch.tensor([[0, 1, 1], 
                   [1, 2, 1], 
                   [2, 0, 1], 
                   [2, 3, 1], 
                   [2, 3, 2]])

Q = Quiver([Q0, Q1])

class TestQuiver(test.TestCase):

    def test_arrows(self):
        arr = Q.arrows()
        self.assertTrue(arr.domain == Q[1])
        self.assertTrue(arr.data.numel() == len(Q1))

    def test_vertices(self):
        vtx = Q[0].range()
        self.assertTrue(vtx.domain == Q[0])
        self.assertTrue(vtx.data.numel() == len(Q0))

#--- Functorial values ---

class SomeFunctor: 

    def __call__(self, i):
        """ 
        Shape at vertex i. 
        """
        return [3]
    
    def fmap(self, a):
        """
        Graph of index map between vertices a[0] and a[1]. 
        
        Starts at src and adds the edge label mod 6.
        """
        src, tgt = a[0], a[1]
        b = a[2]
        return lambda x: (src + (b * x)) % self(tgt)[0]


F = SomeFunctor()

FQ = Quiver([Q0, Q1], functor=F)

class TestQuiverFunctor(test.TestCase):

    def test_objects(self):
        """ objects of functorial quiver """
        obj = FQ[0].range()
        self.assertTrue(FQ[0] == obj.domain)
        self.assertEqual(3 * len(Q0), obj.data.numel())

    def test_arrows(self):
        """ arrows of functorial quiver """
        arr = FQ.arrows()
        self.assertTrue(FQ[1] == arr.domain)
        self.assertEqual(3 * len(Q1), arr.data.numel())

G0 = [0, 1]
G1 = [[0, 1], [1, 2]]
G2 = [[0, 1, 2]]
G = Graph([G0, G1, G2], FreeFunctor(2))

class TestFunctorQuiver(test.TestCase):
    """ Quiver GF.quiver() associated to a functor-valued graph GF 
    
                012 .--> 01 -.--> 0
                     `-> 12 ---`-> 1
    """

    def test_quiver(self):
        """ quiver construction """
        Q = G.quiver()          #|  2**3  | 2**2  | 2**1 |
        self.assertEqual(Q[0].size, 8     + 4 * 2 + 2 * 2)
        self.assertEqual(Q[1].size, 8 * 4 + 4 * 3)

    def test_arrows(self):
        """ arrows of a functorial graph """
        Q = G.quiver()
        F = Q.arrows()
        T2 = fp.Torus([2, 2])
        result = F.data[:Q[1].sizes[0]] 
        expect = (T2.res(0) @ T2.coords)(torch.arange(4))
        self.assertClose(expect.data.flatten(), result.data)

    def test_fmap(self):
        """ subgraphs of the functor """
        Q = G.quiver()
        # u : 01 --> 0
        Fu = Q.fmap([2, 0])
        Fui = Q[0].begin[2] + torch.arange(4)
        Fuj = torch.arange(2).repeat_interleave(2)
        self.assertClose(Fu, torch.stack([Fui, Fuj]))
        # v : 01 --> 1
        Fv = Q.fmap([2, 1])
        Fvi = Fui
        Fvj = Q[0].begin[1] + torch.arange(2).repeat(2)
        self.assertClose(Fv, torch.stack([Fvi, Fvj]))
        # u ++ v ++ (w : 012 --> 12)
        Fuvw = Q.fmap([[2, 2, 4], [0, 1, 3]])
        Fwi = Q[0].begin[4] + torch.arange(8)
        Fwj = Q[0].begin[3] + torch.arange(4).repeat(2)
        Fw = torch.stack([Fwi, Fwj])
        self.assertClose(Fuvw, torch.cat([Fu, Fv, Fw], dim=1))


