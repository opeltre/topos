# Statistics and Topology 

```python
from system import System
from field import Field

K = System(('i:j:k', 'i:k:l', 'j:k:l'))

# Degree-0 field: tensors indexed by faces a
f = K.gaussian(0)

# Degree-1 field: tensors indexed by pairs a > b
phi = K.gaussian(1)

# Usage
g = f + K.delta[1] @ phi
assert g.degree == 0

f1 = K.mu @ K.zeta @ f
assert (f1 - f).trim(1e-6) == 0
``` 

See [zeta.py](zeta.py) for examples. 
