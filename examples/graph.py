from topos import Graph
import topos
import fp
import torch

# vertices
G0 = [0, 1, 2]
# edges
G1 = [[0, 1], [1, 2]]
# graph
G = Graph([G0, G1])

x = G.ones(0)
f = G.ones(1)

N = G.nerve()
phi = N.ones(1)

F = topos.FreeFunctor(3)
GF = Graph(G, F)