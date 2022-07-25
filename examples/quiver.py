import torch
import topos
import fp

from topos import Quiver

""" 
Quiver of symmetric figure eight:

        1 -. .- 2
    Q   :   0   :
        3 -` `- 4

We assign a label to each arrow identifying the elementary loop
it belongs to (1, 2, 3 or 4). 

    fk : i -> j     <=>     fk = [i, j, k]

Arrows are sorted by lexicographic order on their source and target
indices by the Quiver constructor.

    Q[0] = [0, 1, 2, 3, 4]
    Q[1] = [[0, 1, 1], [0, 2, 2], ..., [4, 0, 2], [4, 2, 4]]
"""
# arrow flips
flip = lambda x: torch.stack([x[:,1], x[:,0], x[:,2] + 2], dim=1)

# vertices
Q0 = torch.arange(5)
# arrows
Q1 = torch.tensor([[0, 2, 2], [2, 4, 2], [4, 0, 2], 
                   [0, 1, 1], [1, 3, 1], [3, 0, 1]])
Q1 = torch.cat([Q1, flip(Q1)])

# quiver               
Q = Quiver(Q0, Q1)

# arrow indices as integer field
print(f"Q[1].range() :\n{Q[1].range()}")
# gaussian weights on arrows
print(f"Q[1].randn() :\n{Q[1].randn()}") 
# ...
# print(Q[1].zeros())

"""
Nerve of the quiver Q

The nerve N[d] for d > 0 contains length d chains in Q, 

A chain f in N[d] a set of d composable arrows (f1, ..., fd) 
where f1 : i0 -> i1, f1 : i1 -> i2, etc.

For d = 0 one simply defines N[0] as the set of vertices Q[0]. 
Note that N[1] coïncides with Q[1]

N.B:
----
If Q has cycles, calling Q.nerve() will not terminate. 
With cycles, call Q.nerve(k) instead to stop at length k chains.
"""

from topos import Nerve
N = Nerve.classify(Q, 3)

print(f"N[2].randn() :\n{N[2].range()}")

"""
Functor valued quivers.

A functor `F : Q -> Shape` should define:
- an object map `F(i)` yielding a shape for all i in Q[0] 
- an arrow map `F.fmap(f)` for every edge f in Q[1].

A pair (F0, F1) of callables could define a functor F by default.
""" 

class SomeFunctor: 

    def __call__(self, i):
        """ 
        Shape at vertex i. 
        """
        return [6]
    
    def fmap(self, a):
        """
        Graph of index map between vertices a[0] and a[1]. 
        
        Starts at src and adds the edge label mod 6.
        """
        src, tgt = a[0], a[1]
        b = a[2]
        return (src + (b * torch.arange((self(src)[0])))) % self(tgt)[0]

F = SomeFunctor()

# Quiver with functorial coefficients
FQ = Quiver(Q0, Q1, functor=F)

# indices of FQ[0] = prod [Fi for i in Q[0]]
print(f"FQ[0].range() :\n{FQ[0].range()}")

# field on FQ[1] representing functorial maps Fi -> Fj
print(f"FQ.arrows() :\n{FQ.arrows()}")