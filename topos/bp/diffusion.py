from topos.core import Smooth,VectorField


class Diffusion(VectorField):

    def __new__(cls, N, beta=1):

        DiffN = super().__new__(cls, N[0])
        DiffN.network = N
        
        def retract(self):
            beta = self.beta if 'beta' in dir(self) else 1
            N = self.network
            F = lambda x: N.from_scalars(N.freeEnergy(beta)(x))
            return Smooth(N[0], N[0])(lambda H : H - F(H))

        setattr(DiffN, "retract", retract)

        DiffN.__name__ = f'Diffusion {N}'
        return DiffN

    @classmethod
    def integrator(cls, method):
        def with_retraction(self, dt):
            r = self.retract()
            step = method(self, dt)
            return lambda x: r(step(x))
        return VectorField.integrator(with_retraction)


def BetheDiffusion(network, beta=1):
    """ Bethe diffusion on hamiltonians. """
    N = network
    delta = N.codiff_zeta(1)
    D = N.freeDiff(beta)
    
    @Diffusion(N, beta)
    def diffusion(H):
        return (-1) * delta(D(H))

    return diffusion
   

def GBPDiffusion(network, beta=1):
    """ GBP diffusion on hamiltonians. """
    N = network
    delta, D = N.codiff(1), N.freeDiff(beta)
    zeta = N.zeta(0)

    @Diffusion(N, beta)
    def diffusion(H):
        return (-1) * zeta(delta(D(H)))

    return diffusion