from topos import Graph
import fp

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

from topos import Linear

mat = Linear(G, G).randn()