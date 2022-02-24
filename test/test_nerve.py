import test
import torch

from topos.base import Graph, Nerve

G = Graph([0, 1, 2, 3],
         [[0, 1],[0, 2], [0, 3]],
         [[0, 1, 2], [0, 2, 3], [0, 1, 3]])

N = Nerve.nerve(G)

class TestComplex (test.TestCase):

    def test_zeta(self):
        zeta = N.zeta(0)
        # zt0(x)_a = \sum_{b <= a} x_b
        one = torch.ones([10])
        result = zeta @ one
        expect = torch.tensor([1, 1, 1, 1, 3, 3, 3, 6, 6, 6])
        self.assertClose(expect, result)
