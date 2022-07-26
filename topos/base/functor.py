import fp
import topos.io as io

import torch

class Functor:

    def __init__(self, f0, f1):
        """ Create functor from object and arrow maps. """
        self.obj_map = f0
        self.hom_map = f1

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


class FreeFunctor(Functor):
    """
    Free functor on a hypergraph.

    A region a is mapped to the cartesian product Fa 
    of shapes Fi for i vertex of a.

    An inclusion b <= a is assumed represented by a labeled arrow 
    f = [a, b, 1+i0, ..., i+ik, 0, ..., 0] where i0 < ... < ik < len(a)
    are the position of forgotten vertices from a to b.

    See Graph.quiver() to compute such labeled arows. 
    """

    def __init__(self, graph, shape):
        """ 
        Create a free functor from atomic degrees of freedom.

        The `shape` argument can be an int, a mapping or a callable.
        """
        self.graph = graph
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
            face = graph.coords(a)
            return fp.Torus([F(i) for i in face])

        # arrow map
        def fmap(f):
            f = io.readTensor(f)
            src = obj(f[0])
            tgt = obj(f[1])
            a = graph.coords(f[0]).contiguous()
            b = graph.coords(f[1]).contiguous()
            js = torch.bucketize(b, a)
            return tgt.index @ src.res(*js) @ src.coords 

        super().__init__(obj, fmap)