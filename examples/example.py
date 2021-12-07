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

        p["k"] == p["i:k"].sum(dim=[0])    

    The differential d measures local inconsistencies:
   
        d: Field(K[0]) -> Field(K[1]) -> ... 
"""
dp = K.d(p)                     
dp['i:k > k']   == p['k'] - p['i:k'].sum(dim=[0])
dp.degree       == 1

""" N.B: Uniform beliefs are always consistent: """

K.d @ K.uniform(0) == K.zeros(1)


"""------------ Conservation ---------------------------

    Transport changes potentials without altering
    the total energy the system. 
    
    A potential u is a transport of h if and only if 
    there exists a 1-field phi such that 

        u = h + delta(phi) 

    where delta is the adjoint of d. 
    
    This implies that 
        
        d(u) @ phi == u @ delta(phi) 

    and in particular that Ker(d) vanishes on Im(delta),

    See: Peltre, Message-Passing Algorithms and Homology (2020)
    [theorem 2.14](https://arxiv.org/pdf/2009.11631.pdf#page=48)
"""
K.uniform(0) @ K.delta(K.randn(1)) == 0


"""------------ Diffusion -----------------------------------

        By transporting energy with a regularising heat flux, 
        local Gibbs states may converge to a consistent field
        of local probabilities. 

        The effective energy gradient is a non-linear operator 
        the difference of local hamiltonians with conditional 
        free energies of surrounding regions.
"""

def diffusion (u, eps=1, nit=10) : 
    for k in range(nit):
        U   = K.zeta(u)         # Effective energy field 
        Phi = K.Deff(U)         # Effective energy gradient 
        phi = -eps * K.mu(Phi)  # Bethe heat flux (1-Möbius inversion)
        u  += K.delta(phi)      # Energy conservation
        print(Phi.norm())

diffusion(h)
