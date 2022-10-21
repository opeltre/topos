import matplotlib.pyplot as plt
import torch

def plot_graph(G, x=None, r=.3, c0='#52b', c1='#Cab', size=(10, 10)):
    """ Plot graph using laplacian eigenmodes 1 and 2. """
    G = G.scalars()
    fig = plt.figure(figsize=size)
    plt.axis('equal')
    plt.axis('off')
    if isinstance(c0, str): c0 = [c0] * G.Nvtx
    if isinstance(c1, str): c1 = [c1] * len(G[1].keys)
    #-- eigenmodes --
    if type(x) == type(None):
        L = G.codiff(1) @ G.diff(0)
        L = L.data.to_dense()
        eigval, eigvec = torch.linalg.eigh(L)
        x = eigvec[:,1:3] * G.Nvtx
    #-- arrows --
    for p, ep in enumerate(G[1].keys):
        i, j = ep
        xi, xj = x[i], x[j]
        add_arrow(xi, xj, r, .1, c1[p])
    #-- vertices --
    for i, xi in enumerate(x):
        add_vertex(xi, r, c0[i], label=str(i))           
    return fig

def add_vertex(x, r, c, label=None):
    """ Add vertex at x in current plot. """
    circle = plt.Circle((x[0], x[1]), r, color=c)
    plt.gca().add_patch(circle)
    if label:
        plt.text(x[0], x[1], label, ha='center', va='center', color='#fff')

def add_arrow(xi, xj, r, w, c, label=None):
    """ Add arrow from xi to xj in current plot. """
    v = xj - xi
    dv = r * v / v.norm()
    pos = [*(xi + dv), *(v - 2 * dv)]
    plt.arrow(*pos, 
              width=.1,
              length_includes_head=True,
              head_width=.3,
              color=c,
              head_length=.6)

    

