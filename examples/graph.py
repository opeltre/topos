from topos import Graph

# vertices
G0 = [0, 1, 2]
# edges
G1 = [[0, 1], [1, 2]]
# graph
G = Graph([G0, G1])

x = G.ones(0)
f = G.ones(1)