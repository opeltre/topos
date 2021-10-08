from topos import Sheaf, Domain
from topos.domain.cartesian import Sum, Product

E = Sheaf({'a': [2],  'b': [2]})
F = Sheaf({'i': [3],  'j': [4]})

#--- Sum ---

S = Sum(E, F)

#--- Product ---

P = Product(E, F)

