from topos import System
import torch

K = System.closure(("i:k", "j:k", "k:l"))


"""----------- Parameters --------------------------

        p[a](xa) = exp(-H[a](xa)) / Za  
                 where
                   H[a](xa) = Sum{b <= a}  h[b](xb)
                
"""
h = K.randn(0)      # Local potentials: 0-Field of local tensors
H = K.zeta(h)       # Local hamiltonians (sums of inner potentials)
p = K.gibbs(H)      # Local beliefs (local Gibbs states)



"""----------- Consistency -------------------------
   
    Local beliefs are consistent if and only if 
    for all inclusion 'i:k > k'
""" 
p["k"] == p["i:k"].sum(dim=[0])    

""" The differential d measures local inconsistencies:
   
        d: Field(K[0]) -> Field(K[1]) -> ... 
"""
dp = K.d(p)                     
dp['i:k > k']   == p['k'] - p['i:k'].sum(dim=[0])
dp.degree       == 1

""" Uniform beliefs are always consistent: """

K.d @ K[0].uniform() == K[1].zeros()



"""------------ Transport ---------------------------
   
   
"""

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
