from test import TestCase

import torch
from topos.base import Sheaf

F = Sheaf.sparse([4, 4], [[0, 1], [1, 2], [2, 3], [0, 2]])

class TestSheafSparse(TestCase):

    def test_sparse (self):
        # F.idx is a sparse torch.LongTensor
        self.assertTrue(isinstance(F.idx, torch.sparse.LongTensor))
        # F.index(keys)
        keys = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        result = F.index(keys)
        expect = torch.tensor([0, 1])
        self.assertClose(expect, result)
