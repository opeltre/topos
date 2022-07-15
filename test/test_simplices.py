import unittest
import torch

from topos.core import simplices

class TestSimplices(unittest.TestCase):

    def test_simplices(self):
        F = torch.arange(8).view([2, 4])
        K = simplices(F)
        choices = 4
        for d, Kd in enumerate(K):
            result = tuple(Kd.shape)
            expect = (2, choices, d + 1)
            self.assertEqual(expect, result)
            choices = int(choices * ((4 - (d + 1)) / (d + 2)))

    def test_simplices_indices(self):
        F = torch.arange(12).view([3, 4])
        K, J = simplices(F, indices=True)
        n = F.shape[-1]
        # loop over degrees
        for Kd, Jd in zip(K[:-1], J[:-1]):
            Kd_t = Kd.transpose(0, 1)
            # loop over forgotten indices
            for faces, js in zip(Kd_t, Jd):
                idx = [i for i in range(n) if i not in [int(j) for j in js]]
                idx = torch.tensor(idx, dtype=torch.long)
                equal = (faces == F.index_select(-1, idx))
                self.assertTrue(equal.prod())

