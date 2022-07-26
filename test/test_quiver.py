from topos import Quiver

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
        obj = FQ[0].range()
        self.assertTrue(FQ[0] == obj.domain)
        self.assertEqual(3 * len(Q0), obj.data.numel())

    def test_arrows(self):
        arr = FQ.arrows()
        self.assertTrue(FQ[1] == arr.domain)
        self.assertEqual(3 * len(Q1), arr.data.numel())