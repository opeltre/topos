import topos
import topos.bp as bp

import torch

#--- Warning: G1 has to be sorted ---

G0 = [0, 1, 2, 3, 4]
G1 = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]]

G = topos.Complex([G0, G1], topos.FreeFunctor(2), sort=True)

N = bp.IsingNetwork.classify(G)

GBP = bp.GBPDiffusion(N)

#--- Logs ---

log = {"DH": []}

#@GBP.writer
def log_freeDiff(x, n):
    log["DH"].append(N.freeDiff(x).norm())

#--- Belief Propagation --- 

H0 = N.randn(0)
H1 = GBP.euler(.3, 30)(H0)
p = N.gibbs(H1)

#--- Nerve slices : patch ---

def get_slice(js, degree):
    idx1 = N.index(js) - N.scalars().begin[degree]
    begin = N[degree].begin[idx1]
    end   = N[degree].end[idx1]
    return begin, end 

#--- Singularities ---

def loop_couplings(p):
    """ Return product of edge eigenvalues across loops. """
    G = N._classified 
    eigvals = N.edge_eigvals(p)
    #--- loop indices ---
    loop1 = G.Index([[0, 1], [0, 2], [1, 2]]) - G.Nvtx
    loop2 = G.Index([[0, 3], [0, 4], [3, 4]]) - G.Nvtx
    #--- loop couplings ---
    C1 = eigvals.data[loop1].prod()
    C2 = eigvals.data[loop2].prod()
    return torch.stack([C1, C2])

def is_singular(p, tol=1e-6):
    """ 
    Check if consistent beliefs are singular.  

    The tangent space of consistent potentials intersects 
    the image of the codifferential (gauge) if and only if 
    `p` is singular (see eigen_flux). 

    Tests if loop couplings cancel the singular polynomial. 
    """
    C1, C2 = loop_couplings(p)
    pol = (C1 + 1/3) * (C2 + 1/3) - 4/9   
    return bool(pol.abs() < tol)

def eigen_flux(p):
    """
    Return the flux associated to the largest eigenvalue of 
    the "Kirchhoff" operator linearising diffusion on heat
    fluxes at the neighbourhood of `p`, i.e.

        M(phi) = Lambda . phi 

    where Lambda = 1 if and only if `p` is singular. 
    """
    C1, C2 = loop_couplings(p)
    Delta = (C2 - C1)**2 + 16 * C1 * C2
    a2 = (C2 - C1 + Delta.sqrt()) / (4 * C1)
    b1, b2 = (1 + 2 * a2), (2 + a2)

    eigvecs = N.node_eigvecs(p)
    eigvals = N.edge_eigvals(p)

    if not Delta > 0: 
        raise RuntimeError(f"Delta: {Delta} < 0")
    
    outer = [1, 2, 3, 4]
    phi   = N.zeros(1)

    #--- [0, i] -> 0 ---
    a = torch.tensor([1., 1., a2, a2])
    vec_0 = eigvecs.data[:2]
    for i, ai in zip(outer, a):
        edge_idx  = G.scalars().index([0, i])
        begin, end = get_slice([edge_idx, 0], degree=1)
        phi_i0 = ai * vec_0
        phi.data[begin:end] = phi_i0

    #--- [0, i] -> i ---
    b = torch.tensor([b1, b1, b2, b2])
    for i, bi in zip(outer, b):
        edge_idx  = G.scalars().index([0, i])
        begin, end = get_slice([edge_idx, i], degree=1)
        val_0i = eigvals.data[edge_idx - G.Nvtx]
        vec_i = eigvecs.data[2*(i-1):2*i]
        phi_0i = bi * val_0i * vec_i
        phi.data[begin:end] = phi_0i

    #--- [i, j] -> j ---
    Gsc = G.scalars()
    for ij, bij in zip([(1, 2), (3, 4)], [b1, b2]):
        i, j = ij
        edges = Gsc.index([[i, j], [0, i], [0, j]]) - G.Nvtx
        vals = eigvals.data[edges]
        c_ij = bij * vals[0] * vals[1]
        c_ji = bij * vals[0] * vals[2]
        vec_i = eigvecs.data[2*(i-1):2*i]
        vec_j = eigvecs.data[2*(j-1):2*j]
        phi_ij = c_ij * vec_j
        phi_ji = c_ji * vec_i
        key_ij = G.index([i, j])
        begin_i, end_i = get_slice([key_ij, i], 1)
        begin_j, end_j = get_slice([key_ij, j], 1)
        phi.data[begin_i:end_i] = phi_ji
        phi.data[begin_j:end_j] = phi_ij

    return phi    


#--- Plot graph --- 

def plot_graph():
    """ Plot graph """
    x0 = torch.tensor([0, -1, -1, 1, 1])
    x1 = torch.tensor([0, -.7, .7, -.7, .7])
    topos.io.plot_graph(G, 3 * torch.stack([x0, x1], 1).float())
    topos.io.plt.show()