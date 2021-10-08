from .hashable import Hashable

class Shape :

    def __init__(self, *ns):
        self.dim = len(ns)
        self.n = list(ns)
        size = 1
        for ni in ns:
            size *= ni
        self.size = size

    def index(self, *js):
        if not len(js):
            return 0
        i = js[0]
        for d in range(1, min(len(js), self.dim)):
            i = (i * self.n[d]) + js[d]
        return i

    def coords(self, i):
        if i >= self.size or i < 0:
            raise IndexError(f"{self} coords {i}")
        js = []
        r, div = i, self.size
        for d in range(0, self.dim): 
            div //= self.n[d]
            q, r = divmod(r, div)
            js += [q]
        return js

    def p(self, d):
        def proj_d(i):
            x = self.coords(i)
            return x[d]
        return proj_d

    def res(self, *ds):
        def res_ds(i):
            x   = self.coords(i)
            tgt = Shape([self.n[d] for d in ds])
            return tgt.index(*[x[d] for d in ds])
        return res_ds

    def __iter__(self):
        return self.n.__iter__()
    
    def __str__(self): 
        return "(" + ",".join([str(ni) for ni in self.n]) + ")"

    def __repr__(self): 
        return f"Shape {self}"

