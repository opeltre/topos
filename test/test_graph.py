import test
import torch

from topos.base import Graph

G = Graph([[[0], [1], [2]],
           [[0, 1], [1, 2]],
           [[0, 1, 2]]])

class TestGraph (test.TestCase):

    def test_adj(self):
        # A0
        A0 = torch.ones([3])
        self.assertClose(A0, G.adj[0].to_dense())
        # A1
        A1 = torch.tensor([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        self.assertClose(A1, G.adj[1].to_dense())
        # A2
        A2 = torch.zeros([3, 3, 3])
        A2[0, 1, 2] += 1
        self.assertClose(A2, G.adj[2].to_dense())
