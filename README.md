# Statistics and Topology 

```python
from system import System
from field import Field

K = Hypergraph(('i:j:k', 'i:k:l', 'j:k:l'))

# Degree-0 field: tensors indexed by faces a
f = Field(K, 0, "gaussian") 

# Degree-1 field: tensors indexed by pairs a > b
phi = Field(K, 1, "gaussian")

# Usage
g = f + K.delta(phi)
assert g.degree == 0

h = f.mu().zeta()
assert (h - f).norm() < 1e-6:w
``` 
