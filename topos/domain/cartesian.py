from .sheaf import Sheaf
from .domain import Domain
from topos.base import Shape, Fiber, Sequence, Chain, Seq
from topos.core import Field
from topos.core import Linear, GradedLinear,\
                       Functional, GradedFunctional

from itertools import product

#--- Unit Object ---

class Point (Sheaf): 
    """ Point Domain spanning the field of scalars. """

    def __init__(self, degree=0):
        super().__init__(['\u2022'], None, degree)


#--- Null Object ---

class Empty (Sheaf):
    """ Empty Domain spanning the null vector space {0}. """
    
    def __init__(self, degree=0):
        super().__init__([], None, degree)

    def field(self, data=None, degree=0):
        return super().field([0.], self.degree)


#--- Product ---

class Product (Sheaf):
    """ Cartesian product of domains. """

    def __init__(self, *sheaves): 
        self.grades  = list(sheaves)
        self.rank    = len(sheaves) - 1
        self.trivial = not sum((not f.trivial for f in sheaves))
        self.scalars = self if self.trivial else\
                       self.__class__(*(f.scalars for f in sheaves))
    
        #--- Product of fibers --- 
        def key(*fibers):
            return Sequence.read([f.key for f in fibers])

        def shape(*fibers):
            out = []
            for f in fibers:
                out += f.shape.n
            return out
        shape = {
            key(*fs): shape(*fs) for fs in product(*sheaves)
        }
        super().__init__(shape, ftype=Sequence)

    def __getitem__(self, d=0):
        return self if d == None else\
               self.grades[d]
   
    def projection(self, d):
        def res_d (key):
            return key[d]
        def res_fmap(fiber):
            shape = Shape(*[self[j].get(kj).size\
                           for j, kj in enumerate(fiber.key)])
            return shape.p(d)
        return res_d, res_fmap

    def j(self, d):
        p, pmap = self.projection(d)
        return self[d].pull(self, p, "p*", fmap=pmap)

    def p(self, d):
        return self.j(d).t()
    
    def get(self, key, *keys):
        if isinstance(key, Sequence):
            return self.fibers[key]
        return super().get("|".join([key, *keys]))

#--- Coproduct: disjoint Unions --- 

class Sum (Domain):
    """ Disjoint union of domains. """

    def __init__(self, *sheaves, keymap=None, sort=False):
        self.grades  = list(sheaves)
        self.rank    = len(sheaves) - 1

        def is_trivial(F):
            return F.trivial if isinstance(F, Sheaf) else True

        def scalars(F):
            return F.scalars if "scalars" in F.__dir__() else F

        self.trivial = not sum(not is_trivial(F) for F in sheaves)

        if "scalars" not in self.__dir__() and not self.trivial:
            self.scalars = (
                self.__class__(*(scalars(F) for F in sheaves))
                if "_scalars" not in self.__dir__() 
                else self._scalars()
            )

        #--- Join fibers ---
        keymap = (lambda i, a: Seq.read((i, a))) if not keymap \
                 else keymap
        shape = {
            keymap(i, a): fa.shape for i, Fi in enumerate(sheaves) \
                                   for a, fa in Fi.fibers.items()} 
        keys = list(shape.keys())
        if sort: keys.sort(key=sort)
        shape = shape #if not self.trivial else None
        super().__init__(keys, shape, ftype=Sequence)
        
    def __getitem__(self, d=0):
        return self if d == None else\
               self.grades[d]

    def get(self, key, d=None):
        if d == None:
            return self.fibers[Sequence.read(key)]
        return self[d].get(key)
    
    #--- Subdomain ---
    
    def restriction(self, Ks):
        """ Restriction to a subset of keys. """
        return self.__class__(*
            (self[d].restriction(Kd) for d, Kd in enumerate(Ks))
        )

    #--- Universal projections / embeddings --- 

    def embedding(self, d):
        def emb_d (k): return  Seq.read((d, k))
        return emb_d

    def p (self, d):
        return self.pull(self[d], self.embedding(d), name="j*")
    
    def j (self, d):
        return self.p(d).t()

    def pull(self, src, g=None, name="map*", fmap=None):
        if not isinstance(src, Sum):
            return super().pull(src, g, name, fmap)
        gs = [self[i].pull(si, g[i] if g!= None else g, name)\
                     for i, si in enumerate(src.grades)]
        return GradedLinear([self, src], gs, 0, name)
    
    #--- Lifts --- 

    def lift(self, f, name="name", src=None, tgt=None):
        src = getattr(self, src) if src else self
        tgt = getattr(self, tgt) if tgt else self
        fs = [getattr(Ni, f) for Ni in self.grades] 
        return GradedLinear([src, tgt], fs, 0, name)\
            if isinstance(fs[0], Linear)\
            else GradedFunctional([src, tgt], fs, 0, name=name)
   
    #--- Fields --- 

    def field(self, data, d=None):
        return Field(self, data, d) if d == None\
                else self[d].field(data, d)


#--- Union : no prefix & possible collision of keys

class Union (Sum):

    def __init__(self, *sheaves, sort=False, ftype=Fiber):
        super().__init__(*sheaves, keymap=lambda i, a: a, sort=sort)
        self.ftype = ftype

    def __getitem__(self, key):
        a = self.ftype.read(key)
        return self.fibers[a]

    def field(self, data, d=None):
        return Field(self, data, d) 

#--- Fibration : indexed coproducts

class Fibration (Sum):

    def __init__(self, mapping, sort=False):
        keys = list(mapping.keys())
        if sort: keys.sort(key=sort)
        sheaves = [mapping[k] for k in keys]
        keymap  = lambda i, a: Chain(keys[i], a)
        self._scalars = lambda : self.__class__({
            k: Ak.scalars for k, Ak in mapping.items()
        }, sort)
        super().__init__(*sheaves, keymap=keymap)
        self.mapping = mapping
    
    def __getitem__(self, d=None):
        return self if d == None else self.mapping[d]

    def get(self, key, d=None):
        return self.fibers[Chain.read(key)] if d == None\
               else self[d].get(key)
