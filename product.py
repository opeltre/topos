from mixin import Mappable, Hashable
from set import MapMixin 

class ProductMixin (MapMixin) :

    def __or__(self, other): 
        return self.__class__(xi for xi, i in (*self, *other))

    def project(self, dim=(0,)):
        return self.restrict(dim)

    def p(self, *args):
        return self.project(*args)

    def flip(self, dim=(1, 0)):
        k = len(dim)
        return self.__class__(
            (self[dim[i]] if i < k else xi for xi, i in self))

    def fmap(self, f):
        return self.__class__((f(xi) for xi, i in self))

    def map(self, f):
        return self.__class__((f(xi, i) for xi, i in self))

    def __repr__(self): 
        s = str(self)
        return f"\u03a0-{s}"

    def __str__(self):
        s = ""
        for xi, i in self:
            si = str(xi)
            if '\n' in si: 
                si = ('\n' + si).replace('\n', '\n   ') 
                s += f"\n{i} :{si}"
            else:
                s += f" . {si}" if len(s) > 0 else si
        return s

class Word (ProductMixin, Hashable, tuple): 

    def __iter__(self): 
        return ((xi, i) for i, xi in enumerate(super().__iter__()))
    
    
