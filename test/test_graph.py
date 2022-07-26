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

    def test_index(self):
        # Query 0
        Q0 = torch.arange(4).view([4, 1])
        idx, mask = G.index(Q0, True)
        self.assertClose(idx[:3], torch.arange(3))
        self.assertTrue((mask == torch.tensor([1, 1, 1, 0])).prod())
        # Query 1
        Q1 = torch.tensor([[0, 1], [1, 2], [2, 1], [2, 3]])
        idx, mask = G.index(Q1, True)
        self.assertTrue((mask == torch.tensor([1, 1, 0, 0])).prod())
        self.assertClose(idx[:2], 3 + torch.arange(2))
        # Query 2
        Q2 = torch.tensor([[0, 1, 2], [1, 2, 3]])
        idx, mask = G.index(Q2, True)
        self.assertTrue(idx[0] == 5)
        self.assertTrue((mask == torch.tensor([1, 0])).prod())

    def test_coords(self):
        result = (G.coords(0), G.coords(3), G.coords(5))
        expect = ([0], [0, 1], [0, 1, 2])
        for e, r in zip(expect, result):
            self.assertEqual(e, r.long().tolist())

    def test_quiver(self):
        Q = G.quiver()
        Q0, Q1 = Q[0].keys[:,0].tolist(), Q[1].keys.tolist()
        self.assertEqual(Q0, [0, 1, 2, 3, 4, 5])
        self.assertEqual(Q1, [[3, 0], [3, 1], [4, 1], [4, 2],
                              [5, 0], [5, 1], [5, 2], [5, 3], [5, 4]])