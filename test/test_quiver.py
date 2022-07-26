from topos import Quiver, FreeFunctor, Graph

import torch
import unittest

Q0 = torch.arange(4)
Q1 = torch.tensor([[0, 1, 1], 
                   [1, 2, 1], 
                   [2, 0, 1], 
                   [2, 3, 1], 
                   [2, 3, 2]])

Q = Quiver([Q0, Q1])



class TestQuiver(unittest.TestCase):

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

class TestQuiverFunctor(unittest.TestCase):

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

class TestFunctorQuiver(unittest.TestCase):
    """ Quiver GF.quiver() associated to a functorial graph GF 
    
                012 ---> 01 -.--> 0
                    `-> 12 --`-> 1
    """
    
    def test_quiver(self):
        """ quiver construction """
        Q = G.quiver()
        self.assertEqual(Q[0].size, 8     + 4 * 2 + 2 * 2)
        self.assertEqual(Q[1].size, 8 * 4 + 4 * 3)



