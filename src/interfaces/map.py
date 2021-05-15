class Map: 

    def fmap(self, f):
        return self.__class__({k: f(x) for x, k in self})

    def map(self, f):
        return self.__class__({k: f(x, k) for x, k in self})

    def fibers(self, f=None):
        F = {}
        if f == None:
            for y, x in self: 
                F[y] = F[y] + [x] if y in F else [x]
            return self.__class__(F) 
        for y, x in self: 
            z = f(y, x)
            if z in F:
                F[z][x] = y
            else:
                F[z] = {x: y} 
        return self.__class__(F).fmap(self.__class__)

    def uncurry(self): 
        g = {}
        for fx, x in self:
            for fxy, y in fx: 
                g[(x, y)] = fxy
        return self.__class__(g)

    def curry(self): 
        f = {}
        for gxy, (x, y) in self:
            if x in f:
                f[x][y] = gxy
            else:
                f[x] = {y: gxy}
        return self.__class__(f).fmap(self.__class__)
    
    def restrict(self, dim):
        dim = (dim, ) if type(dim) == int else dim
        return self.__class__((self[i] for i in dim))

    def forget(self, dim):
        dim = (dim, ) if type(dim) == int else dim
        n = len(self)
        dim = [i % n for i in dim]
        axes = [j for j in range(n) if j not in dim]
        return self.restrict(axes)
