from test import TestCase

import torch
from topos.base import Sheaf

F = Sheaf.sparse([4, 4], [[0, 1], [1, 2], [2, 3], [0, 2]])

class TestSheafSparse(TestCase):

    def test_index(self):
        # F.idx is a sparse torch.LongTensor
        self.assertTrue(isinstance(F.idx, torch.sparse.LongTensor))
        # F.index(keys)
        keys = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=torch.long)
        result = F.index(keys)
        expect = torch.tensor([0, 1, 2, 3])
        self.assertClose(expect, result)

    def test_slice(self):
        begin, end, fiber = F.slice([0, 1])
        result = begin, end, fiber.size
        expect = (0, 1, 1)
        self.assertEqual(expect, result)

    def test_slices(self):
        result = len(list(F.slices()))
        expect = 4
        self.assertEqual(expect, result)
