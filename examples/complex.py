import topos
import torch
import fp

from topos import Complex

K = Complex.simplicial([[0, 1, 2], [0, 1, 3], [0, 2, 3]])

F  = topos.FreeFunctor(3)
KF = topos.Complex(K, F)
