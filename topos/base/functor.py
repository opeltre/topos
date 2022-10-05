import fp
import topos.io as io

import torch

class Functor:

    def __init__(self, f0, f1):
        """ Create functor from object and arrow maps. """
        self.obj_map = f0
        self.hom_map = f1
        try:
            if f0.__doc__: self.__call__.__func__.__doc__ = f0.__doc__
            if f1.__doc__:self.fmap.__func__.__doc__ = f1.__doc__ 
        except:
            pass

    def __call__(self, i):
        """ Object map. """
        return self.obj_map(i)
    
    def fmap(self, f):
        """ Arrow map. """
        return self.hom_map(f)

    def __matmul__(self, other):
        """ Functor composition. """
        f0 = lambda i:self(other(i))
        f1 = lambda f:self.fmap(other.fmap(f))
        return Functor(f0, f1)


class ConstantFunctor(Functor):
    """
    Constant Functor i.e. with one target object and identities. 
    """
    def __init__(self, shape):
        self.shape = shape
        super().__init__(lambda _: shape, lambda _:lambda x: x)
    

class FreeFunctor(Functor):
    """
    Free functor on the powerset of integers.

    A region a = [i0, ..., ik] is mapped to the cartesian product: 
    
        F(a) = Fi0 x ... x Fik

    An inclusion b <= a is here assumed represented by a pair
        
        (a, b) = ([i0, ..., ik], [j0, ..., jr]).

    and it is mapped to the restriction that forgets variables in a - b:

        F.fmap((a, b)) : F(a) -> F(b)

    See `Graph.quiver()` to compute inclusions and associated restrictions.
    """

    def __init__(self, shape):
        """ 
        Create a free functor from atomic degrees of freedom.

        The `shape` argument can be an int, a mapping or a callable.
        """
        if not callable(shape) and '__getitem__' in dir(shape):
            F = shape.__getitem__
            self.atomic = F
        elif isinstance(shape, int):
            self.atomic = shape
            F = lambda i: shape
        elif callable(shape):
            F = shape
            self.atomic = F
        assert(callable(F))

        # object map
        def obj(a):
            """ 
            Cartesian product of objects over a = [i0, ..., ik].
            """
            return fp.Torus([F(i) for i in a])

        # arrow map
        def fmap(f):
            """ 
            Restriction map from dims [i0, ..., ik] to dims [j0, ..., ir]. 
            """
            a, b = io.readTensor(f[0]), io.readTensor(f[1])
            src, tgt = obj(a), obj(b)
            js = torch.bucketize(b, a)
            return tgt.index @ src.res(*js) @ src.coords 

        super().__init__(obj, fmap)