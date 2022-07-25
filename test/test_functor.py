from topos import Graph, Functor, FreeFunctor

import torch
import unittest
import fp

G = Graph([[0, 1, 2], 
           [[0, 1], [1, 2]],
           [[0, 1, 2]]])

class TestFunctor(unittest.TestCase):

    def test_constant_free(self):
        F = FreeFunctor(G, 2)
        T1, T3 = fp.Torus([2]), fp.Torus([2, 2, 2])
        result = F(5)
        self.assertEqual(T3, result)
        result = F.fmap([5, 0])
        expect = (T1.index @ T3.res(0) @ T3.coords)(torch.arange(8))
    
    def test_free(self):
        pass