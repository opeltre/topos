import fp

from .field  import Field
from .linear import Linear

from contextlib import contextmanager

class Smooth (fp.Arrow):
    
    def __new__(cls, A, B):

        Fa, Fb = Field(A), Field(B)

        class SmoothAB (fp.Arrow(Fa, Fb)):

            src = Fa
            tgt = Fb

            def __new__(cls, f, df=None, name='\u03bb'):
                map = object.__new__(cls)
                cls.__init__(map, f, df, name)
                return map

            def __init__(self, f, df=None, name='\u03bb'):
                super().__init__(f)
                self.__name__ = name
                if callable(df):
                    self.tangent = fp.Arrow(Field(A), Linear(A, B))

        return SmoothAB
    
    @classmethod
    def name(cls, A, B):
        rep = lambda D: D.__name__ if '__name__' in dir(D) else D.size
        return f'Smooth {rep(A)} -> {rep(B)}'
    
    @classmethod
    def source_type(cls, f, xs):
        assert(len(xs) == 1)
        x = xs[0]
        s_x, s_in = tuple(x.shape), tuple(f.src.shape)
        if s_x == s_in: 
            return f.src
        elif s_x[-len(s_in):] == s_in: 
            return f.src.batched(*s_x[:-len(s_in)])

    @classmethod
    def target_type(cls, f, xs):
        x = xs[0]
        s_x = tuple(x.shape)
        s_in, s_out = tuple(f.src.shape), tuple(f.tgt.shape)
        if s_x == s_in:
            return f.tgt
        elif s_x[-len(s_in):] == s_in:
            return f.tgt.batched(*s_x[:-len(s_in)])

class VectorField:

    def __new__(cls, A):

        class VecA(Smooth(A, A)):
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._writers = []
                self._log = []
            
            @cls.integrator
            def euler(self, dt):
                """ Explicit Euler integrator. """
                return lambda x: x + dt * self(x)

            @contextmanager
            def write(self, **callbacks):
                out = {k: [] for k in callbacks}
                self._writers = ([
                    (lambda x, _: out[k].append(fk(x)))
                                for k, fk in callbacks.items()])
                try:
                    yield out
                finally:
                    self._writers = []

            def writer(self, callback):
                """ 
                Use as decorator to append an iteration callback.

                The writer will be called as `callback(x, n)` where:
                    - `x` is the vector field argument,
                    - `n` is the current iteration number. 
                """
                self._writers.append(callback)
                return callback
        
        VecA.__name__ = f'VectorField {A}'
        return VecA

    def __pow__(self, n):
        def fn(x):
            y = x
            for i in range(n):
                y = self(y)
            return y
        return self.__class__(fn)

    @classmethod   
    def integrator(cls, method):

        def loop_method(self, dt, n=1):

            src = self.src.domain

            step = Smooth(src, src)(method(self, dt))

            @Smooth(src, src)
            def integrate(x0):
                x = x0 + 0
                for k in range(n):
                    #--- writers ---
                    if len(self._writers):
                        for w in self._writers:
                            w(x, n)
                    #--- step ---
                    x = step(x)
                return x
            integrate.__name__ = f'{method.__name__} {dt} x {n}'
            return integrate

        return loop_method