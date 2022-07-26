from topos import Graph, Functor, FreeFunctor

import torch
import test
import fp

G = Graph([[0, 1, 2], 
           [[0, 1], [1, 2]],
           [[0, 1, 2]]])

class TestFunctor(test.TestCase):

    def test_constant_free(self):
        """ Constant atomic shapes (binary variables) """
        F = FreeFunctor(2)
        T1, T3 = fp.Torus([2]), fp.Torus([2, 2, 2])
        # object map
        result = F([0, 1, 2])
        self.assertEqual(T3, result)
        # morphism map
        idx = torch.arange(8)
        result = F.fmap([[0, 1, 2], [0]])(idx)
        expect = (T1.index @ T3.res(0) @ T3.coords)(idx)
        self.assertClose(expect.data, result.data)
    
    def test_free(self):
        """ Variable atomic degrees of freedom """
        F = FreeFunctor(lambda i: int(2 + i))
        # object map
        T12 = fp.Torus([3, 4])
        T012 = fp.Torus([2, 3, 4])
        self.assertEqual(F([1, 2]), T12)
        self.assertEqual(F([0, 1, 2]), T012)
        # morphism map
        idx = torch.arange(2 * 3 * 4)
        result = F.fmap([[0, 1, 2], [1, 2]])(idx)
        expect = (T12.index @ T012.res(1, 2) @ T012.coords)(idx)
        self.assertClose(expect.data, result.data)