from topos.base import Hypergraph, Set, Chain
from topos import System
import torch

from topos.core.operators import face, differential, codifferential
from topos import Linear

K = System.closure(("i:j", "j:k", "k:l"))
# K = System(("i:j", "j:k", "k:l"))
# K = System(("i:j:k", "i:k:l", "j:k:l"))

# : Potential 
u = K[0].randn() 
# : Local energy 
U = K.zeta(u)
# : Belief (local Gibbs state)
p = K[0].gibbs(u)

#--- differential : (in)consistency ---
dp = K.d(p)

#--- codifferential : energy transport ---

def diffusion (u, eps=1, nit=10) : 
    for k in range(nit):
        U   = K.zeta(u)
        #   : Effective energy gradient 
        Phi = K.Deff(U)
        #   : (Bethe) Heat flux
        phi = -eps * K.mu(Phi)
        #   : Energy conservation
        u  += K.delta(phi) 
        print(Phi.norm())

def project(A, B):
    pull = A.pull(B)
