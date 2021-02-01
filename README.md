# Statistics and Topology 

```
K = Hypergraph(('i:j:k', 'i:k:l', 'j:k:l'))
K.close()

f = K.field(0, "gaussian") 
g = f.d().delta() + f.delta().d()

print(g.degree)

h = f.mu().zeta()

print((h - f).norm())
``` 
