from topos.base import Sheaf
import test

import torch

F = Sheaf(['a', 'b'], [[3, 2], [2, 3]])

class TestSheaf(test.TestCase):

    def test_ones(self):
        result = F.ones().data
        expect = torch.ones([12])
        self.assertClose(expect, result)

    def test_slice(self):
        begin, end, fiber = F.slice('b')
        result = (begin, end, tuple(fiber.shape))
        expect = (6, 12, (2, 3))
        self.assertTrue(expect == result)

    def test_getitem(self):
        fiber = F['a']
        result = fiber.shape
        expect = [3, 2]
        self.assertTrue(expect == result)
