# Topos

## Algebra and Topology

Multilinear algebra tensors `t` can be viewed as collections of scalars `t[i|j|...|k]`
indexed by words in the alphabet of coordinates `i, j, k ...`. 

The tensor product `t | s` of two tensors is then defined by `(t | s)[a|b] = t[a] * s[b]` for every pair of words `a` and `b` concatenated as `a | b`. 

The monoidal structure `|` on words can be chosen universal or free. It can an also be symmetric, yielding polynomial algebras in its tensor powers. It can be antisymmetric, yielding the exterior algebras of lengths, surfaces, volumes... in geometry. A less common example is the case were a partial order `a > b` is enforced. Differential operators `d` and `delta = d.t()` also act on the algebra of tensors, induced by elementary k-face maps acting on the length-n word `i0|...|in` by removing `ik`. 

## Statistics 

This library is concerned with a more general case where each tensor component `t[a] = t[i|j|...|k]` is not a scalar but a tensor of shape `(Ni, ..., Nk)` instead. Such tensors naturally occur in statistics, where each `t[i|j|...|k]` represents a function or probability density on a cartesian product of `Ni * ... * Nk` elements. 

Keys `a = i|j|...|k` then correspond to subsets of simultaneously observed variables. The physical range of interactions, our concurrent measurement capacity of the size of our memories are all very different practical reasons for limiting words to a certain admissible size.   


```python
from system import System
from field import Field

K = System(('i:j', 'j:k'))

# Degree-0 field: tensors indexed by faces a
f = K.gaussian(0)
# Tensor {
#   (i:j) :->   [[* , *],
#                [* , *]]
#   
#   (j:k) :->   [[* , *],
#                [* , *]] 
#
#   (j) :->     [* , *]       
# }

# Degree-1 field: tensors indexed by pairs a > b
phi = K.gaussian(1)
# Tensor {
#   (i:j) . (j) :-> [* , *]
#   (j:k) . (j) :-> [* , *]
# }

# Codifferential delta : A[n] <- A[n + 1]
g = f + K.delta[1] @ phi
assert g.degree == 0

# Combinatorial operations zeta and mu
f1 = K.mu @ K.zeta @ f
assert (f1 - f).trim(1e-6) == 0
``` 

See [zeta.py](zeta.py) for examples. 
