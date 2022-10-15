from topos.core import VectorField


def codiff_zeta(network, d=1):
    N = network
    return N.zeta(d-1) @ N.codiff(d) @ N.mu(d)

def BetheDiffusion(N, beta=1):
    """ Bethe diffusion on hamiltonians. """
    delta = codiff_zeta(N, 1)
    D = N.freeDiff(beta)
    F = N.freeEnergy(beta)
    j = N.from_scalars(0)

    @VectorField(N[0])
    def diffusion(H):
        return (-1) * (delta(D(H)) + j(F(H)))

    return diffusion

def BetheDiffusion_mu(network, beta=1):
    """ Bethe diffusion on potentials. """
    N = network
    delta, D = N.codiff(1), N.freeDiff(beta)
    zeta, mu = N.zeta(0), N.mu(1)
    F = N.freeEnergy(beta)

    @VectorField(N[0])
    def diffusion(h):
        return (-1) * (
            delta(mu(D(zeta(h))))
          + N.mu((N.from_scalars(F(zeta(h)))))
        )
    
    return diffusion

def GBPDiffusion(network, beta=1):
    """ GBP diffusion on potentials. """
    N = network
    delta, D = N.codiff(1), N.freeDiff(beta)
    zeta = N.zeta(0)

    @VectorField(N[0])
    def diffusion(h):
        return (-1) * (
            delta(D(zeta(h)))
          + N.mu(N.from_scalars(N.freeEnergy(beta)(N.zeta(h))))
        )    
    return diffusion