import topos
import topos.bp as bp

import torch
import matplotlib.pyplot as plt

#--- Warning: G1 has to be sorted ---

G0 = [0, 1, 2, 3, 4]
G1 = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]]

G = topos.Complex([G0, G1], topos.FreeFunctor(2), sort=True)

N = bp.IsingNetwork.classify(G)

GBP = bp.GBPDiffusion(N)

#--- Logs ---

log = {"DH": []}

@GBP.writer
def log_freeDiff(x, n):
    log["DH"].append(N.freeDiff(x).norm())

def clear_logs():
    log["H"], log["DH"] = [], []

#--- Belief Propagation --- 

H0 = N.randn(0)
H1 = GBP.euler(.3, 30)(H0)
p = N.gibbs(H1)

#--- Nerve slices : patch ---

def get_slice(js, degree):
    """
    Helper: return slice associated to a nerve hyperedge. 
    """
    idx1 = N.index(js) - N.scalars().begin[degree]
    begin = N[degree].begin[idx1]
    end   = N[degree].end[idx1]
    return begin, end 

#--- Energy/temperature diagrams ---

def initial_conditions(field=0, interval=(0, 2, 63), eps=1):
    #--- singularity
    p = singular_beliefs(field)
    phi = eigen_flux(p)
    dH = N.zeta(0) @ N.codiff(1) @ phi
    #--- initial energies
    H = N._ln(p)
    betas = torch.linspace(*interval)
    H0 = [beta * H for beta in betas]
    H1 = [beta * H + eps * dH for beta in betas]
    H2 = [beta * H - eps * dH for beta in betas]
    return (H0, H1, H2)

def energy_temperature(field=0, interval=(0, 2, 63), eps=1, dt=.3, n=200):
    Hs = initial_conditions(field, interval, eps)
    h0 = N.mu(N._ln(singular_beliefs(field)))
    U = [[], [], []]
    for i, His in enumerate(Hs): 
        for H in His: 
            Hf = GBP.euler(dt, n)(H)
            pf = N.gibbs(Hf)
            energies = N.to_scalars(pf * h0)
            U[i].append(energies.data.sum())
    return U

#--- Plot trajectory --- 

def plot_singular_freeDiff(b=0, c=None, eps=.1, dt=.2, n=100):
    #--- initial condition
    p   = singular_beliefs(b, c)
    phi = eigen_flux(p)
    dH  = N.zeta(N.codiff(phi))
    H0 = N._ln(p) + eps * (dH / dH.norm())
    #--- diffusion
    clear_logs() 
    H1 = GBP.euler(dt, n)(H0)
    #--- plot inconsistency
    plt.plot(torch.stack(log["DH"]), color="orange")
    plt.title("Free Energy Differences")
    plt.show()

#--- Singularities ---

def singular_interaction(fields=0, couplings=None):
    """
    Construct singular interaction from local fields and couplings. 
    """
    if type(couplings) == type(None):
        C = torch.tensor([1/3])
        logC = torch.log(C)
        s_ij = (-logC/3).repeat(6)
    if isinstance(fields, (int, float)):
        s_i = torch.tensor([fields]).repeat(5)
    interaction = torch.cat([s_i, s_ij])
    return N.scalars().Field(0)(interaction)

def singular_beliefs(fields=0, couplings=None):
    """
    Construct singular beliefs from local fields and couplings. 
    """
    return N.lift_interaction(singular_interaction(fields, couplings))

def loop_couplings(p):
    """
    Return product of edge eigenvalues across loops. 
    """
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

    Test if loop couplings cancel the determinant of linearised diffusion. 

    The tangent space of consistent potentials intersects 
    the image of the codifferential (space of gauge choices) 
    if and only if `p` is singular. See `eigen_flux(p)` for 
    a generator of the singular intersection. 
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