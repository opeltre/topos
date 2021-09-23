from topos import Hypergraph, Set, Chain
from topos import System
import torch

from topos.core.operators import face, differential, codifferential
from topos.core.functional import Matrix

K = System(("i:j", "j:k"))
# K = System(("i:j", "j:k", "k:l"))
# K = System(("i:j:k", "i:k:l", "j:k:l"))

mat_d = differential(K, 0)

d = Matrix(mat_d, 1, "d")
delta = Matrix(mat_d.t(), -1, "d*")
L = d @ delta

u = K.randn() 
du = d(u)
