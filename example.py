from topos import Hypergraph, Set, Chain
from topos import System
import torch

from topos.core.operators import face, differential, codifferential
from topos import Matrix

K = System(("i:j", "j:k"))
# K = System(("i:j", "j:k", "k:l"))
# K = System(("i:j:k", "i:k:l", "j:k:l"))

#--- differential ---

u = K[0].randn() 
du = K.d[0] @ u

#--- codifferential ---

phi = K[1].randn()
dt_phi = K.delta[1] @ phi

#--- laplacian ---

L0, L1 = K.delta[1] @ K.d[0], K.d[0] @ K.delta[1]

#--- Gibbs density

exp = K[0].map(torch.exp)
e_u = exp(-u)
rho = K[0].gibbs(u)

#--- zeta ---

h = K[0].ones()
H = K.zeta[0] @ h
