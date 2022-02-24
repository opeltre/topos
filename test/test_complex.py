import test
import torch

from topos.base import Complex

K = Complex.simplicial([[0, 1, 2, 3], [1, 2, 3, 4]])

class TestComplex (test.TestCase):

    def test_diff(self):
        # d0 : K0 -> K1 vanishes on constant fields
        d0 = K.diff(0)
        result = d0 @ torch.ones([5])
        expect = torch.zeros([9])
        self.assertClose(expect, result)
        # d1 : K1 -> K2
        d1 = K.diff(1)
        self.assertTrue(list(d1.shape) == [7, 9])
        # d1 @ d0 = 0
        result = torch.sparse.mm(d1, d0).to_dense()
        expect = torch.zeros([7, 5]) 
        self.assertClose(expect, result)
        # d2 : K2 -> K3
        d2 = K.diff(2)
        self.assertTrue(list(d2.shape) == [2, 7])
        # d2 @ d1 = 0
        result = torch.sparse.mm(d2, d1).to_dense()
        expect = torch.zeros([2, 9])
        self.assertClose(expect, result)
