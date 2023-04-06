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
    log["DH"].append(N.freeDiff(x).norm(dim=[-1]))

def clear_logs():
    log["H"], log["DH"] = [], []

#--- Belief Propagation --- 

#H0 = N.Field(0).batch([N.randn(0) for i in range(2000)])
#H1 = GBP.euler(.3, 100)(H0)
#p = N.gibbs(H1)

def main():
    traces = torch.stack(log['DH']).T
    for t in traces[:10]:
        plt.plot(t)

class Eight(bp.IsingNetwork):

    def __new__(cls, *args, **kwargs):
        if not len(args):
            G0 = [0, 1, 2, 3, 4]
            G1 = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]]
            G = topos.Complex([G0, G1], topos.FreeFunctor(2), sort=True)
            return cls.classify(G)
        else: 
            N = super().__new__(cls)
            N.__init__(*args, **kwargs)
            return N

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            super().__init__(*args, **kwargs)

    def energy(N, u, v):
        return N.to_scalars(0)(u * v).data.sum([-1])

    #--- Nerve slices : patch ---

    def get_slice(N, js, degree):
        """
        Helper: return slice associated to a nerve hyperedge. 
        """
        idx1 = N.index(js) - N.scalars().begin[degree]
        begin = N[degree].begin[idx1]
        end   = N[degree].end[idx1]
        return begin, end 

    #--- Energy/temperature diagrams ---

    def initial_conditions(N, field=0, beta=1, eps=1, coupling=None):
        """
        Initial conditions 
        """
        #--- singularity
        p = N.singular_beliefs(field, coupling)
        phi = N.eigen_flux(p)
        dH = N.zeta(0) @ N.codiff(1) @ phi
        #--- initial energies
        H = N._ln(p)
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta)
        slc = (*([slice(None)] * beta.dim()), 
                *([None] * H.dim()))
        H0 = beta[slc] * H.data 
        if H.data.dim() > 1 and beta.dim():
            H0 = H0.transpose(0, 1)
        return N[0].field(H0)

    def energy_temperature(N, field=0, interval=(0, 2, 63), eps=1, dt=.3, n=200):
        Hs = N.initial_conditions(field, interval, eps)
        h0 = N.mu(N._ln(N.singular_beliefs(field)))
        U = [[], [], []]
        for i, His in enumerate(Hs): 
            for H in His: 
                Hf = GBP.euler(dt, n)(H)
                pf = N.gibbs(Hf)
                energies = N.to_scalars(pf * h0)
                U[i].append(energies.data.sum())
        return U

    #--- Singularities ---

    def singular_interaction(N, fields=0, couplings=None):
        """
        Construct singular interaction from local fields and couplings. 
        """
        s_i = (fields if isinstance(fields, torch.Tensor)
                      else torch.tensor(fields))
        if s_i.dim() == 0:
            s_i = s_i.repeat(5)
        elif not s_i.shape[-1] == 5:
            s_i = s_i.unsqueeze(-1).repeat(*([1] * s_i.dim()), 5)
        if couplings is not None:
            s_ij = torch.tensor(couplings)
        else:
            C = torch.tensor([1/3])
            logC = torch.log(C)
            shape = (*s_i.shape[:-1], 6)
            s_ij = (-logC/3).repeat(*shape)
        interaction = torch.cat([s_i, s_ij], -1)
        return N.scalars().field(interaction, 0)

    def singular_beliefs(N, fields=0, couplings=None):
        """
        Construct singular beliefs from local fields and couplings. 
        """
        return N.lift_interaction(N.singular_interaction(fields, couplings))

    def loop_couplings(N, p):
        """
        Return product of edge eigenvalues across loops. 
        """
        G = N._classified 
        eigvals = N.edge_eigvals(p)
        #--- loop indices ---
        loop1 = G.Index([[0, 1], [0, 2], [1, 2]]) - G.Nvtx
        loop2 = G.Index([[0, 3], [0, 4], [3, 4]]) - G.Nvtx
        #--- loop couplings ---
        slc = [slice(None)] * (eigvals.data.dim() - 1)
        C1 = eigvals.data[(*slc, loop1)].prod(-1)
        C2 = eigvals.data[(*slc, loop2)].prod(-1)
        return torch.stack([C1, C2], -1)

    def is_singular(N, p, tol=1e-6):
        """ 
        Check if consistent beliefs are singular.  

        Test if loop couplings cancel the determinant of linearised diffusion. 

        The tangent space of consistent potentials intersects 
        the image of the codifferential (space of gauge choices) 
        if and only if `p` is singular. See `eigen_flux(p)` for 
        a generator of the singular intersection. 
        """
        C1, C2 = N.loop_couplings(p)
        pol = (C1 + 1/3) * (C2 + 1/3) - 4/9   
        return bool(pol.abs() < tol)

    def eigen_flux(N, p):
        """
        Return the flux associated to the largest eigenvalue of 
        the "Kirchhoff" operator linearising diffusion on heat
        fluxes at the neighbourhood of `p`, i.e.

            M(phi) = Lambda . phi 

        where Lambda = 1 if and only if `p` is singular. 
        """
        C1, C2 = N.loop_couplings(p).transpose(0, -1)
        Delta = (C2 - C1)**2 + 16 * C1 * C2
        a2 = (C2 - C1 + Delta.sqrt()) / (4 * C1)
        b1, b2 = (1 + 2 * a2), (2 + a2)

        eigvecs = N.node_eigvecs(p).data.transpose(0, -1)
        eigvals = N.edge_eigvals(p).data.transpose(0, -1)

        if not (Delta > 0).prod(): 
            raise RuntimeError(f"Delta: {Delta} < 0")
        
        #--- shapes --- 
        n1 =  N.sizes[1]
        ns = p.shape[:-1]
        
        outer = [1, 2, 3, 4]
        #--- accumulator
        phi = torch.zeros(*ns, n1).transpose(0, -1)

        #--- [0, i] -> 0 ---
        one = torch.ones(tuple(a2.shape))
        a   = torch.stack([one, one, a2, a2])
        vec_0 = eigvecs.data[:2]
        for i, ai in zip(outer, a):
            edge_idx  = G.scalars().index([0, i])
            begin, end = N.get_slice([edge_idx, 0], degree=1)
            phi_i0 = ai * vec_0
            phi.data[begin:end] = phi_i0

        #--- [0, i] -> i ---
        b = torch.stack([b1, b1, b2, b2])
        for i, bi in zip(outer, b):
            edge_idx  = G.scalars().index([0, i])
            begin, end = N.get_slice([edge_idx, i], degree=1)
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
            begin_i, end_i = N.get_slice([key_ij, i], 1)
            begin_j, end_j = N.get_slice([key_ij, j], 1)
            phi.data[begin_i:end_i] = phi_ji
            phi.data[begin_j:end_j] = phi_ij

        return N[1].field(phi.transpose(0, -1))


#--- Plot graph --- 

def plot_graph():
    """ Plot graph """
    x0 = torch.tensor([0, -1, -1, 1, 1])
    x1 = torch.tensor([0, -.7, .7, -.7, .7])
    topos.io.plot_graph(G, 3 * torch.stack([x0, x1], 1).float())
    topos.io.plt.show()

 #--- Plot trajectory --- 

def plot_singular_freeDiff(N, b=0, c=None, eps=.1, dt=.2, n=100):
    #--- initial condition
    p   = N.singular_beliefs(b, c)
    phi = N.eigen_flux(p)
    dH  = N.zeta(N.codiff(phi))
    H0 = N._ln(p) + eps * (dH / dH.norm())
    #--- diffusion
    clear_logs() 
    H1 = GBP.euler(dt, n)(H0)
    #--- plot inconsistency
    plt.plot(torch.stack(log["DH"]), color="orange")
    plt.title("Free Energy Differences")
    plt.show()
