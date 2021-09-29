from topos import System

""" BP on an acyclic graph """

# Do not include the empty region 
K = System(("i:j", "j:k", "j:l", "l:m", "m:n"), void=False)

# Local potentials 
u = K[0].randn()

# Local hamiltonians
U = K.zeta[0](u)

""" Algorithm """

T, DU_norm = 10, []
for t in range(T):

    # effective energy gradient
    phi  = - K.Deff(U)

    # transport of potentials 
    u   += K.delta[1](phi)

    # conservation of total energy
    U    = K.zeta[0](u)

    #--- view trajectory ---
    DU_norm += [phi.norm()]

print(DU_norm)
