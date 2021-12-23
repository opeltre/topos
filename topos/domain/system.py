from .nerve import Nerve
from .sheaf import Sheaf, Simplicial

from topos.base import Hypergraph, Shape
from topos.core import Functional, Linear, GradedLinear
from topos.core import sparse
from topos.core.operators import coface, nabla, zeta_chains,\
                                 interaction, embed_interaction

import torch

class System (Nerve): 
    """ 
    Simplicial nerve N[d](K, E) of a free sheaf E over K.

    The local shape of region `a` in K =~ N[0](K, E) is given by:

        E[a] = (E[i] for i in a)
    
    i.e. E[a] is the cartesian product of the E[i]'s. 
    Each chain `a0 > ... > ad` in N[d](K, E) is then assigned a shape:
        
        E[a0 > ... > ad] = E[ad]

    """
    @classmethod
    def closure(cls, K, shape=2, degree=-1, void=1, free=True):
        """ Closure for `cap`. """
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        K = K.closure() 
        if not void: 
            K = Hypergraph((r for r in K if len(r) > 0))
        return cls(K, shape, degree, free)

    def __init__(self, K, shape=2, degree=-1, sort=1, free=True):
        """ Simplicial complex on the nerve of a hypergraph K. """

        #--- Compute Nerve ---
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        self.hypergraph = K
        nerve = K.nerve(degree, sort=sort)
        self.rank = len(nerve) - 1

        #--- Scalar Fields ---
        self.trivial = (shape == None)
        if not self.trivial:
            N  = [Simplicial(Nk, None, degree=k)\
                           for k, Nk in enumerate(nerve)]
            self.scalars = Nerve(*N)

        #--- Local microstates --- 
        if type(shape) == int:
            E = lambda i : shape
        elif callable(shape):
            E = lambda i : shape(i)
        elif isinstance(shape, dict):
            E = lambda i : shape[i]

        self.microstates = Sheaf({
            i: [E(i)] for i in K.vertices().list()
        })

        self.vertices = self.microstates.scalars
        
        #--- Nerve fibers --- 
        NE = lambda c: Shape(*[E(i) for i in c[-1].list()])
        nerve = [Simplicial(Nk, NE, degree=k)\
                       for k, Nk in enumerate(nerve)]

        super().__init__(*nerve)

        #--- Effective Energy gradient --- 

        if self.rank >= 1:
            d0 = coface(self, 0, 0)
            d1 = coface(self, 0, 1)
            def Deff (U): 
                return d0 @ U + torch.log(d1 @ torch.exp(- U))
            self.Deff = Functional.map([self[0], self[1]], Deff, "\u018a")
        
        #--- Spectral decomposition 

        Z = Nerve(*(Simplicial({
            a: [n-1 for n in Ea.shape] for a, Ea in Kd.items()
        }) for Kd in self.grades))
        
        self.cocycles = Z
        
        #--- Cocycle representation (for measures) ---
        
        resZ = []
        for Zd, Kd in zip(Z.grades, self.grades):
            nat   = lambda a: embed_interaction(Zd[a.key], Kd[a.key])
            resZ += [Kd.pull(Zd, None, "Res Z", nat)]
        ResZ = GradedLinear([self, Z], resZ, 0, "Res Z")
        self.res_Z = ResZ
        self.to_cocycle   = ResZ @ self.fft 
        self.from_cocycle = self.cozeta @ self.ifft @ ResZ.t()

        #--- Coboundary representation (for potentials) ---
        
        Is = []
        pairs = zeta_chains(self, self.rank)
        for Kd, Zd, pairs_d in zip(self.grades, Z.grades, pairs):
            ij = []
            for a, b in pairs_d:
                Zb, Ka = Zd[b], Kd[a]
                ij += interaction(Zb, Ka)
            mat = sparse.matrix([Zd.size, Kd.size], ij)
            Is += [Linear([Kd, Zd], mat, name="I") @ Kd.fft]

        self.interaction = GradedLinear([self, Z], Is, name="I")
        self.from_interaction = self.ifft @ self.res_Z.t() 
    
    def nabla(self, p, degree=0):
        """ Tangent map of Deff at local beliefs p. """ 
        Ds = [nabla(self, d, p.data) for d in range(0, degree + 1)]
        return GradedLinear([self], Ds, 1, "\u2207_p")
   
    def BP (self, beliefs, nit=20, e=0.5, **kwargs):
        """ Generalised belief propagation (BP) algorithm. 
            
            Runs diffusion on the log-likelihoods, 
            and returns the associated Gibbs state. 
            See `K.diffusion` for more info. 
        """
        H0 = self._ln(beliefs)
        H1 = self.diffusion(H0, nit, e, **kwargs)
        return self.gibbs(H1)

    def diffusion (self, energies, nit=20, e=0.5, writer=None, bethe=True):
        """ Diffusion on local hamiltonians (log-likelihoods). 

            Returns consistent local energies when convergent. 
            Consistency can be assessed either on the free energy 
            gradient `K.Deff(H)` or on the associated local beliefs 
            via `K.d(K.gibbs(H))`.
            
            The diffusivity coefficient e (default 0.5) scales 
            heat fluxes, acting as a time step. 

            Parameters:
            -----------
                energies :: Field K[0]           
                nit      :: Int            iterations (20)
                e        :: Float          diffusivity (.5) 

            Returns:
            --------
                energies :: Field K[0]  
        """
        div = self.zeta[0] @ self.delta[1] @ self.mu[1]
        D   = self.Deff
        out = energies + 0.
        for i in range(nit):
            flux = - e * D(out)
            out += div(flux)
        return out


    def dirac (self, x=None, degree=0):
        """ Local dirac masses associated to a global configuration x.
            
            The configuration x is uniformly sampled if not given. 
            It should otherwise be passed as a field on K.vertices.
            
            Parameters:
            -----------
                x       :: Field K.vertices     (None)
                degree  :: Int                  (0)

            Returns:
            --------
                dirac_x :: Field K[degree]

            N.B. `K.dirac(x, 0)` is always a cocycle. In general:
                
                K.d(K.dirac(x, 2n-1)) == K.dirac(2n) * (-1)**n
                K.d(K.dirac(x, 2n))   == K.zeros(2n+1) 
        """
        if isinstance(x, type(None)):
            E = self.microstates
            x = E.scalars.field(torch.cat([
                torch.randint(Ei.shape.size, [1]) for Ei in E
            ]))
        dx = self.zeros(0)
        for a, Ea in self[degree].items():
            dx[a][tuple(x.get(i).long() for i in a[-1].list())] = 1.
        return dx

    def restriction(self, K): 
        """ System restricted to keys in K. """
        K = Hypergraph(K) if not isinstance(K, Hypergraph) else K
        shape = {fiber.key[1] : fiber.shape for fiber in self}
        return self.__class__(K, shape, degree=self.rank, free=False)

    def __repr__(self): 
        return f"System {self}"
